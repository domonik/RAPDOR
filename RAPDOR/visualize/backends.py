import os
from dash import DiskcacheManager, CeleryManager
from RAPDOR.visualize import CONFIG, DISPLAY, DISPLAY_FILE
from dash_extensions.enrich import FileSystemBackend, RedisBackend
from RAPDOR.visualize.dataBackEnd import DisplayModeBackend


if CONFIG["REDIS"]["long_callbacks"]:
    URL = f"redis://{CONFIG['REDIS']['host']}:{CONFIG['REDIS']['port']}/{CONFIG['REDIS']['long_callbacks']}"
    try:
        from celery import Celery
    except ImportError:
        raise ImportError("The optional dependency celery is not installed. Install it via pip install celery[redis]")
    celery_app = Celery(__name__, broker=URL, backend=URL)
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    celery_app = None
    cache = diskcache.Cache(os.path.join(CONFIG["backend"]["name"], "cache"))
    background_callback_manager = DiskcacheManager(cache)

if CONFIG["REDIS"]["database"]:
    data_backend = RedisBackend(host=CONFIG['REDIS']['host'], port=CONFIG['REDIS']['port'], db=CONFIG["REDIS"]["database"])
elif not DISPLAY:
    data_backend = FileSystemBackend(cache_dir=CONFIG["backend"]["name"], threshold=CONFIG["backend"]["threshold"])
else:
    data_backend = DisplayModeBackend(DISPLAY_FILE, cache_dir=CONFIG["backend"]["name"],
                                      threshold=CONFIG["backend"]["threshold"])
