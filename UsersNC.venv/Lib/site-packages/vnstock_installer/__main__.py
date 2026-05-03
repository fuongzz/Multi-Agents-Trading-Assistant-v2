"""
Main entry point for Vnstock Installer
Handles command-line arguments and starts the UI
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

from . import __version__, config, ui, validation


def setup_logging(logging_level):
    """Setup logging configuration"""
    # Configure root logger
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('vnstock_installer')
    logger.info(f"Vnstock Installer v{__version__} starting...")
    return logger


def start_ui(logging_level, build_directory_override):
    """Open the interface"""
    logger = setup_logging(logging_level)
    
    # Suppress the global logger to only show error+ to the console
    logging.getLogger().handlers[1].setLevel(logging.ERROR)
    
    # Setup the build folder
    if build_directory_override is None:
        config.temporary_directory = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {config.temporary_directory}")
    else:
        config.temporary_directory = build_directory_override
        logger.info(f"Using custom build directory: {config.temporary_directory}")
    
    try:
        # Start UI
        logger.info(f"Starting UI with mode: {config.ui_open_mode}")
        ui.start(config.ui_open_mode)
    except (SystemExit, KeyboardInterrupt):
        logger.info("UI closed by user")
        pass
    finally:
        # Remove build folder to clean up from builds (if we created it)
        if build_directory_override is None and config.temporary_directory:
            try:
                shutil.rmtree(config.temporary_directory)
                logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")


def run():
    """Module entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Vnstock Installer - Professional GUI installer for Vnstock packages"
    )
    
    parser.add_argument(
        '-db',
        '--default-browser',
        action='store_true',
        help='use the default browser instead of Chrome/Edge'
    )
    
    parser.add_argument(
        '-nu',
        '--no-ui',
        action='store_true',
        help='do not open a browser, only start the server (useful for debugging)'
    )
    
    parser.add_argument(
        '-c',
        '--config',
        nargs='?',
        type=validation.argparse_file_json,
        help='provide a JSON file containing UI configuration',
        default=None
    )
    
    parser.add_argument(
        '-bdo',
        '--build-directory-override',
        nargs='?',
        help='a directory for build files (overrides the default)',
        default=None
    )
    
    parser.add_argument(
        '-lang',
        '--language',
        nargs='?',
        choices=['vi', 'en'],
        help='hint the language to use by default (vi=Vietnamese, en=English)',
        default=None,
        metavar='LANGUAGE_CODE'
    )
    
    parser.add_argument(
        '--logging-level',
        nargs='?',
        type=validation.argparse_logging_level,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='the level to use for logging - defaults to INFO',
        default='INFO'
    )
    
    parser.add_argument(
        '--base-url',
        nargs='?',
        help=f'Vnstock API base URL (default: {config.DEFAULT_BASE_URL})',
        default=config.DEFAULT_BASE_URL
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'vnstock-installer {__version__}'
    )
    
    args = parser.parse_args()
    
    # Setup config from arguments
    config.supplied_ui_configuration = args.config
    config.language_hint = args.language
    config.base_url = args.base_url
    
    if args.no_ui:
        config.ui_open_mode = config.UIOpenMode.NONE
    elif args.default_browser:
        config.ui_open_mode = config.UIOpenMode.DEFAULT_BROWSER
    else:
        config.ui_open_mode = config.UIOpenMode.CHROME_OR_EDGE
    
    # Validate --build-directory-override exists if supplied
    if (args.build_directory_override is not None) and (
        not os.path.isdir(args.build_directory_override)
    ):
        raise ValueError("--build-directory-override must be a directory")
    
    logging_level = getattr(logging, args.logging_level)
    start_ui(logging_level, args.build_directory_override)


if __name__ == "__main__":
    run()
