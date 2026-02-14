import os
import subprocess
import time
import shutil
import sys
import threading
import datetime
import signal

# --- CONFIGURACIÓN ---
WORK_DIR = r"D:/Video/Video_Rene_Zanelli"
ESRGAN_BIN = r"D:/realesrgan/realesrgan-ncnn-vulkan.exe"

# Modelo y Escala
MODEL_NAME = "realesrgan-x4plus"
SCALE_FACTOR = "4"

# Carpetas
INPUT_DIRS = [
    "INPUT_AI_video_rene_zanelli",
    "INPUT_AI_Zanelli_Ana_Miranda",
    "INPUT_AI_zanelli_2"
]

BATCH_SIZE = 500

# --- NUEVO: carpeta base en NAS para "avance" (OUTPUT) ---
# OJO: el script NO escribe acá. Sólo la usa para detectar frames ya movidos manualmente.
NAS_BASE_DIR = r"Z:/zanelli"

# --- CONFIGURACIÓN DEL MONITOR ---
MONITOR_INTERVAL = 300  # Segundos (5 minutos)
EMA_ALPHA = 0.3

# Tamaño mínimo para considerar un archivo como válido (bytes)
# Una imagen 4x upscaleada de un frame HD debería ser > 100KB
MIN_VALID_FILE_SIZE = 50 * 1024  # 50 KB

# --- ESTADO GLOBAL PROTEGIDO ---
class MonitorState:
    """Estado compartido thread-safe para el monitor."""
    def __init__(self):
        self._lock = threading.Lock()
        self._running = True
        self._current_folder = None
        self._total_frames = 0
        self._output_folder_local = None
        self._output_folder_nas = None
        self._nas_cache_basenames = None  # Cache de basenames válidos en NAS (opcional)
        self._current_batch = []  # Archivos del batch actual siendo procesados

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = value

    def get_state(self):
        """Obtiene una copia atómica del estado actual."""
        with self._lock:
            return {
                'current_folder': self._current_folder,
                'total_frames': self._total_frames,
                'output_folder_local': self._output_folder_local,
                'output_folder_nas': self._output_folder_nas,
                'nas_cache_basenames': set(self._nas_cache_basenames) if self._nas_cache_basenames is not None else None,
                'current_batch': list(self._current_batch)
            }

    def update_folder(self, folder_name, output_folder_local, output_folder_nas, total_frames):
        """Actualiza atómicamente la información de carpeta."""
        with self._lock:
            self._current_folder = folder_name
            self._output_folder_local = output_folder_local
            self._output_folder_nas = output_folder_nas
            self._total_frames = total_frames
            self._current_batch = []

    def set_nas_cache(self, basenames_set):
        """Fija (o limpia) el cache de basenames válidos del NAS para evitar lecturas repetidas."""
        with self._lock:
            self._nas_cache_basenames = set(basenames_set) if basenames_set is not None else None

    def set_current_batch(self, batch_files):
        """Registra los archivos del batch actual."""
        with self._lock:
            self._current_batch = list(batch_files)

    def clear_current_batch(self):
        """Limpia el batch actual (cuando termina exitosamente)."""
        with self._lock:
            self._current_batch = []


monitor_state = MonitorState()

# Proceso externo (para poder terminarlo en cleanup)
current_process = None
process_lock = threading.Lock()


class ProgressDaemon(threading.Thread):
    """Demonio que monitorea el progreso de forma thread-safe."""

    def __init__(self):
        super().__init__()
        self.daemon = True
        self._lock = threading.Lock()
        self._avg_speed = None
        self._last_check_time = time.time()
        self._last_count = 0

    def run(self):
        log("Demonio de monitoreo iniciado.")
        time.sleep(10)

        while monitor_state.running:
            self.check_progress()
            time.sleep(MONITOR_INTERVAL)

    def reset_for_new_folder(self, initial_count):
        """Resetea el estado del monitor para una nueva carpeta."""
        with self._lock:
            self._last_count = initial_count
            self._last_check_time = time.time()
            self._avg_speed = None

    def format_time(self, seconds):
        if seconds is None:
            return "Calculando..."
        return str(datetime.timedelta(seconds=int(seconds)))

    def check_progress(self):
        state = monitor_state.get_state()
        total = state['total_frames']
        local_folder = state['output_folder_local']
        nas_folder = state['output_folder_nas']

        if not local_folder or not os.path.exists(local_folder):
            return

        current_count = count_valid_output_files_union(local_folder, nas_folder, state.get('nas_cache_basenames'))

        with self._lock:
            now = time.time()
            time_delta = now - self._last_check_time
            frame_delta = current_count - self._last_count

            if time_delta <= 0:
                return

            instant_speed = frame_delta / time_delta

            if self._avg_speed is None:
                self._avg_speed = instant_speed if instant_speed > 0 else 0.1
            else:
                self._avg_speed = (EMA_ALPHA * instant_speed) + ((1 - EMA_ALPHA) * self._avg_speed)

            remaining = total - current_count
            if remaining < 0:
                remaining = 0

            percentage = (current_count / total) * 100 if total > 0 else 0
            eta_seconds = remaining / self._avg_speed if self._avg_speed > 0.001 else None

            self._last_check_time = now
            self._last_count = current_count

            avg_speed = self._avg_speed

        print("\n" + "*" * 60)
        print(f" [MONITOR] Reporte de Avance - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f" Carpeta Actual: {state['current_folder']}")
        if nas_folder:
            print(f" Progreso (LOCAL+NAS): {current_count} / {total} frames ({percentage:.2f}%)")
            print(f"   - Local: {count_valid_output_files(local_folder)} válidos")
            print(f"   - NAS:   {count_valid_output_files(nas_folder)} válidos (puede ser más lento)")
        else:
            print(f" Progreso:       {current_count} / {total} frames ({percentage:.2f}%)")
        print(f" Velocidad:      {avg_speed:.2f} frames/seg (Filtro EMA)")
        print(f" Tiempo Restante:{self.format_time(eta_seconds)}")
        print("*" * 60 + "\n")


def count_valid_output_files(folder):
    """Cuenta archivos de salida válidos (tamaño > mínimo)."""
    if not folder or not os.path.exists(folder):
        return 0

    count = 0
    try:
        for entry in os.scandir(folder):
            if entry.name.lower().endswith(('.jpg', '.png')):
                try:
                    if entry.stat().st_size >= MIN_VALID_FILE_SIZE:
                        count += 1
                except OSError:
                    # Archivo siendo escrito, ignorar
                    pass
    except Exception:
        pass
    return count


def get_valid_output_basenames(folder):
    """Obtiene los basenames de archivos válidos en la carpeta dada."""
    valid = set()
    if not folder or not os.path.exists(folder):
        return valid

    try:
        for entry in os.scandir(folder):
            if entry.name.lower().endswith(('.jpg', '.png')):
                try:
                    if entry.stat().st_size >= MIN_VALID_FILE_SIZE:
                        # Normalizar a .png para comparación
                        valid.add(os.path.splitext(entry.name)[0] + ".png")
                except OSError:
                    pass
    except Exception:
        pass
    return valid


def count_valid_output_files_union(local_folder, nas_folder, nas_cache_basenames=None):
    """Cuenta frames válidos únicos considerando LOCAL + NAS (unión por basename).

    Si se entrega nas_cache_basenames, NO se lee el NAS en cada llamada.
    """
    s = set()
    s |= get_valid_output_basenames(local_folder)
    if nas_folder:
        if nas_cache_basenames is not None:
            s |= nas_cache_basenames
        else:
            s |= get_valid_output_basenames(nas_folder)
    return len(s)


def log(msg):
    print(f"[MAIN] {msg}")


def get_missing_frames(src_folder, local_folder, nas_folder, nas_cache_basenames=None):
    """
    Obtiene frames faltantes usando como 'completados' la unión:
      - UPSCALED local (output_folder_local)
      - OUTPUT en NAS (output_folder_nas), movido manualmente por ti
    """
    if not os.path.exists(src_folder):
        return []

    src_files = set(f for f in os.listdir(src_folder) if f.lower().endswith(".png"))

    completed = set()
    completed |= get_valid_output_basenames(local_folder)
    if nas_folder:
        completed |= (nas_cache_basenames if nas_cache_basenames is not None else get_valid_output_basenames(nas_folder))

    missing = src_files - completed
    return sorted(list(missing))


def cleanup_incomplete_files(output_folder, batch_files):
    """
    Elimina archivos incompletos/corruptos del último batch (sólo en salida LOCAL).
    Se llama cuando hay una interrupción.
    """
    if not output_folder or not os.path.exists(output_folder):
        return

    removed = 0
    for filename in batch_files:
        basename = os.path.splitext(filename)[0]

        # Buscar tanto .png como .jpg
        for ext in ['.png', '.jpg']:
            output_path = os.path.join(output_folder, basename + ext)
            if os.path.exists(output_path):
                try:
                    size = os.path.getsize(output_path)
                    if size < MIN_VALID_FILE_SIZE:
                        os.remove(output_path)
                        removed += 1
                        log(f"Eliminado archivo incompleto: {basename + ext} ({size} bytes)")
                except Exception as e:
                    log(f"Error al verificar/eliminar {output_path}: {e}")

    if removed > 0:
        log(f"Se eliminaron {removed} archivos incompletos del último batch.")


def cleanup_staging(staging_dir):
    """Limpia la carpeta de staging."""
    if os.path.exists(staging_dir):
        try:
            for f in os.listdir(staging_dir):
                try:
                    os.remove(os.path.join(staging_dir, f))
                except Exception:
                    pass
            log("Staging limpiado.")
        except Exception as e:
            log(f"Error limpiando staging: {e}")


def _same_drive(path_a, path_b):
    try:
        return os.path.splitdrive(os.path.abspath(path_a))[0].lower() == \
               os.path.splitdrive(os.path.abspath(path_b))[0].lower()
    except Exception:
        return False


def _nas_output_for_local_output(local_output_folder_name):
    """
    Dado un output local tipo 'UPSCALED_xxx', retorna:
      Z:\zanelli\OUTPUT_xxx
    """
    if not NAS_BASE_DIR:
        return None
    nas_name = local_output_folder_name.replace("UPSCALED", "OUTPUT")
    return os.path.join(NAS_BASE_DIR, nas_name)


def graceful_shutdown(signum, frame):
    """Manejador de señal para Ctrl+C."""
    global current_process

    print("\n" + "=" * 60)
    log("¡Interrupción recibida! Limpiando...")

    # Detener el monitor
    monitor_state.running = False

    # Terminar el proceso externo si está corriendo
    with process_lock:
        if current_process and current_process.poll() is None:
            log("Terminando proceso de Real-ESRGAN...")
            current_process.terminate()
            try:
                current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                current_process.kill()

    # Obtener estado actual
    state = monitor_state.get_state()

    # Limpiar staging
    staging_dir = os.path.join(WORK_DIR, "_STAGING_HARDLINKS")
    cleanup_staging(staging_dir)

    # Limpiar archivos incompletos del batch actual (sólo LOCAL)
    if state['current_batch']:
        log(f"Verificando {len(state['current_batch'])} archivos del batch interrumpido...")
        cleanup_incomplete_files(state['output_folder_local'], state['current_batch'])

    print("=" * 60)
    log("Limpieza completada. Puedes reiniciar el script de forma segura.")
    sys.exit(0)


def run_smart_upscaling():
    global current_process

    monitor_thread = ProgressDaemon()
    monitor_thread.start()

    os.chdir(WORK_DIR)

    # Staging persistente (LOCAL)
    staging_dir = os.path.join(WORK_DIR, "_STAGING_HARDLINKS")
    if not os.path.exists(staging_dir):
        os.makedirs(staging_dir)
    else:
        cleanup_staging(staging_dir)

    warned_cross_drive = False

    for input_folder in INPUT_DIRS:
        if not monitor_state.running:
            break

        if not os.path.exists(input_folder):
            continue

        # Salida LOCAL (NO CAMBIAR)
        output_folder_local = input_folder.replace("INPUT_AI", "UPSCALED")
        if not os.path.exists(output_folder_local):
            os.makedirs(output_folder_local)

        # Salida NAS sólo para "avance"
        output_folder_nas = _nas_output_for_local_output(output_folder_local)

        total_src_files = len([f for f in os.listdir(input_folder) if f.lower().endswith(".png")])

        # Actualizar estado de forma atómica
        monitor_state.update_folder(input_folder, output_folder_local, output_folder_nas, total_src_files)

        # Cachear una vez el estado del NAS (si existe) para no re-leerlo en cada chequeo
        nas_cache = get_valid_output_basenames(output_folder_nas) if output_folder_nas else set()
        monitor_state.set_nas_cache(nas_cache)

        # Resetear contador del monitor (usa unión LOCAL+NAS)
        initial_count = count_valid_output_files_union(output_folder_local, output_folder_nas, nas_cache)
        monitor_thread.reset_for_new_folder(initial_count)

        log(f"Analizando {input_folder}...")
        missing_files = get_missing_frames(input_folder, output_folder_local, output_folder_nas, nas_cache)
        total_missing = len(missing_files)

        if total_missing == 0:
            log("Carpeta completada (según LOCAL+NAS). Saltando.")
            continue

        log(f"Faltan {total_missing} frames por procesar (según LOCAL+NAS).")
        if output_folder_nas:
            log(f"NAS (solo lectura para avance): {output_folder_nas}")

        # Procesamiento por lotes
        for i in range(0, total_missing, BATCH_SIZE):
            if not monitor_state.running:
                break

            batch = missing_files[i: i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_missing + BATCH_SIZE - 1) // BATCH_SIZE

            log(f"Procesando batch {batch_num}/{total_batches} ({len(batch)} frames)...")

            # Registrar batch actual para limpieza en caso de interrupción
            monitor_state.set_current_batch(batch)

            # Crear hardlinks en staging (LOCAL)
            for filename in batch:
                src = os.path.join(input_folder, filename)
                dst = os.path.join(staging_dir, filename)

                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except Exception:
                        pass

                try:
                    if _same_drive(src, staging_dir):
                        os.link(src, dst)
                    else:
                        if not warned_cross_drive:
                            log("AVISO: staging y origen en drives distintos. Usando copy2.")
                            warned_cross_drive = True
                        shutil.copy2(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

            # Ejecutar Real-ESRGAN (salida LOCAL)
            cmd = [
                ESRGAN_BIN, "-i", staging_dir, "-o", output_folder_local,
                "-n", MODEL_NAME, "-s", SCALE_FACTOR, "-t", "0",
                "-f", "png", "-g", "0", "-j", "4:4:4"
            ]

            try:
                with process_lock:
                    current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                current_process.wait()

                with process_lock:
                    if current_process.returncode != 0:
                        log(f"Real-ESRGAN terminó con código {current_process.returncode}")
                    current_process = None

            except Exception as e:
                log(f"Error ejecutando Real-ESRGAN: {e}")
                monitor_state.running = False
                break

            # Batch completado exitosamente - limpiar registro
            monitor_state.clear_current_batch()

            # Limpiar staging
            for filename in batch:
                p = os.path.join(staging_dir, filename)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    monitor_state.running = False
    log("Proceso finalizado.")


def verify_output_integrity(output_folder):
    """
    Función de utilidad para verificar integridad de archivos de salida.
    Puede ejecutarse independientemente para limpiar archivos corruptos.
    """
    if not os.path.exists(output_folder):
        return 0

    removed = 0
    for entry in os.scandir(output_folder):
        if entry.name.lower().endswith(('.jpg', '.png')):
            try:
                if entry.stat().st_size < MIN_VALID_FILE_SIZE:
                    os.remove(entry.path)
                    removed += 1
                    print(f"Eliminado: {entry.name}")
            except Exception:
                pass

    return removed


def main():
    # Registrar manejador de señales
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    print("=" * 60)
    print("   UPSCALING CON MONITOR INTELIGENTE (Versión Robusta + NAS avance)")
    print("=" * 60)
    print(f"  Reporte cada {MONITOR_INTERVAL} segundos")
    print(f"  Filtro EMA Alpha: {EMA_ALPHA}")
    print(f"  Tamaño mínimo válido: {MIN_VALID_FILE_SIZE // 1024} KB")
    print(f"  NAS_BASE_DIR (solo lectura): {NAS_BASE_DIR}")
    print(f"  Presiona Ctrl+C para salir de forma segura")
    print("=" * 60 + "\n")

    try:
        run_smart_upscaling()
    except Exception as e:
        log(f"Error inesperado: {e}")
        monitor_state.running = False
        raise


if __name__ == "__main__":
    main()
