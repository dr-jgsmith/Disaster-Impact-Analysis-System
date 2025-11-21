"""
Configuration management for DIAS service.

This module handles loading and validating configuration from environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Settings are loaded from:
    1. Environment variables
    2. .env file (if present)
    3. Default values defined in this class
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================================================================
    # Application Settings
    # ========================================================================
    app_env: str = Field(default="development", description="Application environment")
    app_name: str = Field(default="DIAS", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # ========================================================================
    # API Configuration
    # ========================================================================
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Enable auto-reload")
    
    # CORS
    cors_origins: str = Field(default="*", description="CORS allowed origins")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    # ========================================================================
    # External Services
    # ========================================================================
    google_maps_api_key: Optional[str] = Field(
        default=None,
        description="Google Maps API key for elevation data"
    )
    
    # ========================================================================
    # File Storage
    # ========================================================================
    data_dir: str = Field(default="./data", description="Data directory path")
    output_dir: str = Field(default="./output", description="Output directory path")
    temp_dir: str = Field(default="./tmp", description="Temporary directory path")
    max_upload_size: int = Field(
        default=104857600,
        description="Maximum upload file size in bytes"
    )
    
    @property
    def data_dir_path(self) -> Path:
        """Get data directory as Path object."""
        return Path(self.data_dir).resolve()
    
    @property
    def output_dir_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_dir).resolve()
    
    @property
    def temp_dir_path(self) -> Path:
        """Get temp directory as Path object."""
        return Path(self.temp_dir).resolve()
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    default_impact_range_min: float = Field(
        default=3.0,
        description="Default minimum impact range"
    )
    default_impact_range_max: float = Field(
        default=14.0,
        description="Default maximum impact range"
    )
    default_iterations: int = Field(
        default=500,
        description="Default number of simulation iterations"
    )
    default_time_step: int = Field(
        default=25,
        description="Default simulation time step"
    )
    default_impact_multiplier: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Default impact multiplier"
    )
    default_distance_threshold: float = Field(
        default=100.0,
        description="Default distance threshold for connectivity (meters)"
    )
    
    @property
    def default_impact_range(self) -> Tuple[float, float]:
        """Get default impact range as tuple."""
        return (self.default_impact_range_min, self.default_impact_range_max)
    
    # ========================================================================
    # Performance Settings
    # ========================================================================
    jax_enable_x64: bool = Field(
        default=True,
        description="Enable 64-bit precision in JAX"
    )
    jax_platform_name: str = Field(
        default="cpu",
        description="JAX platform (cpu or gpu)"
    )
    num_threads: int = Field(
        default=4,
        description="Number of threads for parallel operations"
    )
    
    # ========================================================================
    # Monitoring & Logging
    # ========================================================================
    log_file: str = Field(
        default="logs/dias.log",
        description="Log file path"
    )
    log_max_bytes: int = Field(
        default=10485760,
        description="Maximum log file size"
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of log file backups"
    )
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    
    # ========================================================================
    # Security
    # ========================================================================
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    jwt_secret: Optional[str] = Field(
        default=None,
        description="JWT secret for token signing"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    jwt_expiration: int = Field(
        default=3600,
        description="JWT token expiration in seconds"
    )
    
    # ========================================================================
    # Development Settings
    # ========================================================================
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_docs: bool = Field(
        default=True,
        description="Enable API documentation endpoints"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    
    # ========================================================================
    # Validators
    # ========================================================================
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid option."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper
    
    @field_validator("jax_platform_name")
    @classmethod
    def validate_jax_platform(cls, v: str) -> str:
        """Validate JAX platform name."""
        valid_platforms = ["cpu", "gpu"]
        v_lower = v.lower()
        if v_lower not in valid_platforms:
            raise ValueError(f"JAX platform must be one of {valid_platforms}")
        return v_lower
    
    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Validate application environment."""
        valid_envs = ["development", "staging", "production", "testing"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"App environment must be one of {valid_envs}")
        return v_lower
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.data_dir_path,
            self.output_dir_path,
            self.temp_dir_path,
            Path(self.log_file).parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"
    
    def configure_jax(self) -> None:
        """Configure JAX based on settings."""
        os.environ["JAX_ENABLE_X64"] = str(self.jax_enable_x64)
        os.environ["JAX_PLATFORM_NAME"] = self.jax_platform_name
    
    def __str__(self) -> str:
        """Return string representation of settings."""
        return (
            f"DIAS Settings:\n"
            f"  Environment: {self.app_env}\n"
            f"  Version: {self.app_version}\n"
            f"  API: {self.api_host}:{self.api_port}\n"
            f"  Log Level: {self.log_level}\n"
            f"  JAX Platform: {self.jax_platform_name}\n"
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the application settings instance.
    
    This function implements a singleton pattern to ensure settings
    are loaded only once.
    
    Returns:
        Settings instance
    
    Example:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> print(settings.api_port)
        8000
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
        _settings.configure_jax()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Useful for testing or when environment variables change.
    
    Returns:
        New Settings instance
    """
    global _settings
    _settings = None
    return get_settings()


# Export for convenience
settings = get_settings()

