from app.config import get_settings

settings = get_settings()

print(f"App Name: {settings.app_name}")