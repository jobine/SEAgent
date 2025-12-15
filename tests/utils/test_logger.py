"""Unit tests for utils.logger module."""

import io
import os
import logging
import tempfile
import pytest

from src.utils.logger import ColoredFormatter, setup_logging, get_logger, suppress_logging


class TestColoredFormatter:
    """Test cases for ColoredFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a ColoredFormatter instance."""
        return ColoredFormatter('%(levelname)s - %(message)s')

    @pytest.fixture
    def log_record(self):
        """Create a base log record for testing."""
        def _create_record(level: int, msg: str) -> logging.LogRecord:
            return logging.LogRecord(
                name='test',
                level=level,
                pathname='test.py',
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None
            )
        return _create_record

    def test_debug_message_has_blue_color(self, formatter, log_record):
        """Test that DEBUG messages are formatted with blue color."""
        record = log_record(logging.DEBUG, 'debug message')
        result = formatter.format(record)
        
        assert '\033[94m' in result  # Blue color code
        assert '\033[0m' in result   # Reset code
        assert 'DEBUG' in result
        assert 'debug message' in result

    def test_info_message_has_green_color(self, formatter, log_record):
        """Test that INFO messages are formatted with green color."""
        record = log_record(logging.INFO, 'info message')
        result = formatter.format(record)
        
        assert '\033[92m' in result  # Green color code
        assert '\033[0m' in result   # Reset code
        assert 'INFO' in result
        assert 'info message' in result

    def test_warning_message_has_yellow_color(self, formatter, log_record):
        """Test that WARNING messages are formatted with yellow color."""
        record = log_record(logging.WARNING, 'warning message')
        result = formatter.format(record)
        
        assert '\033[93m' in result  # Yellow color code
        assert '\033[0m' in result   # Reset code
        assert 'WARNING' in result
        assert 'warning message' in result

    def test_error_message_has_red_color(self, formatter, log_record):
        """Test that ERROR messages are formatted with red color."""
        record = log_record(logging.ERROR, 'error message')
        result = formatter.format(record)
        
        assert '\033[91m' in result  # Red color code
        assert '\033[0m' in result   # Reset code
        assert 'ERROR' in result
        assert 'error message' in result

    def test_critical_message_has_magenta_color(self, formatter, log_record):
        """Test that CRITICAL messages are formatted with magenta color."""
        record = log_record(logging.CRITICAL, 'critical message')
        result = formatter.format(record)
        
        assert '\033[95m' in result  # Magenta color code
        assert '\033[0m' in result   # Reset code
        assert 'CRITICAL' in result
        assert 'critical message' in result

    def test_unknown_level_uses_reset_color(self, formatter):
        """Test that unknown log levels use reset color."""
        # Create a record with a custom level
        record = logging.LogRecord(
            name='test',
            level=99,  # Custom level
            pathname='test.py',
            lineno=1,
            msg='custom level message',
            args=(),
            exc_info=None
        )
        record.levelname = 'CUSTOM'
        result = formatter.format(record)
        
        # Should still have reset code at the end
        assert '\033[0m' in result
        assert 'custom level message' in result

    def test_formatter_preserves_original_format(self, formatter, log_record):
        """Test that formatter includes the original format components."""
        record = log_record(logging.INFO, 'test message')
        result = formatter.format(record)
        
        # Format is '%(levelname)s - %(message)s'
        assert 'INFO - test message' in result

    def test_colors_dict_contains_all_standard_levels(self):
        """Test that COLORS dict contains all standard log levels."""
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in expected_levels:
            assert level in ColoredFormatter.COLORS

    def test_reset_constant_is_defined(self):
        """Test that RESET constant is properly defined."""
        assert ColoredFormatter.RESET == '\033[0m'


class TestSetupLogging:
    """Test cases for setup_logging function.
    
    Note: These tests manually manage the root logger handlers to avoid
    interference from pytest's logging capture mechanism.
    """

    def _save_and_clear_handlers(self):
        """Save current handlers and clear them."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        root_logger.handlers.clear()
        return root_logger, original_handlers, original_level

    def _restore_handlers(self, root_logger, original_handlers, original_level):
        """Restore original handlers."""
        # Close all current handlers before clearing (important for FileHandler on Windows)
        for h in root_logger.handlers:
            h.close()
        root_logger.handlers.clear()
        for h in original_handlers:
            root_logger.addHandler(h)
        root_logger.setLevel(original_level)

    def test_setup_logging_adds_console_handler(self):
        """Test that setup_logging adds a console handler."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging()
            
            assert len(root_logger.handlers) == 1
            assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_sets_log_level(self):
        """Test that setup_logging sets the specified log level."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging(level=logging.WARNING)
            
            assert root_logger.level == logging.WARNING
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_default_level_is_debug(self):
        """Test that default log level is DEBUG."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging()
            
            assert root_logger.level == logging.DEBUG
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_with_file_handler(self):
        """Test that setup_logging adds file handler when log_file is specified."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_file = os.path.join(tmpdir, 'test.log')
                setup_logging(log_file=log_file)
                
                assert len(root_logger.handlers) == 2
                
                handler_types = [type(h) for h in root_logger.handlers]
                assert logging.StreamHandler in handler_types
                assert logging.FileHandler in handler_types
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for h in root_logger.handlers:
                    h.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates the log directory if it doesn't exist."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_dir = os.path.join(tmpdir, 'nested', 'logs')
                log_file = os.path.join(log_dir, 'test.log')
                
                assert not os.path.exists(log_dir)
                
                setup_logging(log_file=log_file)
                
                assert os.path.exists(log_dir)
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for h in root_logger.handlers:
                    h.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_avoids_duplicate_handlers(self):
        """Test that setup_logging doesn't add handlers if already configured."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging()
            initial_handler_count = len(root_logger.handlers)
            
            # Call setup_logging again - should not add more handlers
            setup_logging()
            
            assert len(root_logger.handlers) == initial_handler_count
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_uses_colored_formatter_for_console(self):
        """Test that console handler uses ColoredFormatter."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging()
            
            console_handler = root_logger.handlers[0]
            assert isinstance(console_handler.formatter, ColoredFormatter)
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_file_handler_uses_standard_formatter(self):
        """Test that file handler uses standard Formatter (not colored)."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_file = os.path.join(tmpdir, 'test.log')
                setup_logging(log_file=log_file)
                
                file_handler = next(
                    h for h in root_logger.handlers 
                    if isinstance(h, logging.FileHandler)
                )
                
                # Should be standard Formatter, not ColoredFormatter
                assert isinstance(file_handler.formatter, logging.Formatter)
                assert not isinstance(file_handler.formatter, ColoredFormatter)
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for h in root_logger.handlers:
                    h.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_setup_logging_writes_to_file(self):
        """Test that log messages are written to file."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_file = os.path.join(tmpdir, 'test.log')
                setup_logging(log_file=log_file)
                
                logger = logging.getLogger('test_write')
                logger.info('Test log message')
                
                # Flush handlers
                for handler in root_logger.handlers:
                    handler.flush()
                
                with open(log_file, 'r') as f:
                    content = f.read()
                
                assert 'Test log message' in content
                assert 'test_write' in content
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for h in root_logger.handlers:
                    h.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger('test_module')
        
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_correct_name(self):
        """Test that get_logger returns logger with correct name."""
        logger = get_logger('my.custom.module')
        
        assert logger.name == 'my.custom.module'

    def test_get_logger_returns_same_instance_for_same_name(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger('same_name_test')
        logger2 = get_logger('same_name_test')
        
        assert logger1 is logger2

    def test_get_logger_returns_different_instances_for_different_names(self):
        """Test that get_logger returns different instances for different names."""
        logger1 = get_logger('name_one_test')
        logger2 = get_logger('name_two_test')
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_with_empty_name(self):
        """Test that get_logger with empty name returns root logger."""
        logger = get_logger('')
        root_logger = logging.getLogger()
        
        assert logger is root_logger

    def test_get_logger_with_hierarchical_name(self):
        """Test that get_logger works with hierarchical names."""
        parent_logger = get_logger('parent_hier_test')
        child_logger = get_logger('parent_hier_test.child')
        grandchild_logger = get_logger('parent_hier_test.child.grandchild')
        
        assert child_logger.parent is parent_logger
        assert grandchild_logger.parent is child_logger


class TestLoggerIntegration:
    """Integration tests for the logger module."""

    def _save_and_clear_handlers(self):
        """Save current handlers and clear them."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        root_logger.handlers.clear()
        return root_logger, original_handlers, original_level

    def _restore_handlers(self, root_logger, original_handlers, original_level):
        """Restore original handlers."""
        # Close all current handlers before clearing (important for FileHandler on Windows)
        for h in root_logger.handlers:
            h.close()
        root_logger.handlers.clear()
        for h in original_handlers:
            root_logger.addHandler(h)
        root_logger.setLevel(original_level)

    def test_full_logging_workflow(self):
        """Test complete logging workflow from setup to logging."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_file = os.path.join(tmpdir, 'app.log')
                
                # Setup logging
                setup_logging(log_file=log_file, level=logging.INFO)
                
                # Get loggers for different modules
                logger1 = get_logger('module.one.integ_test')
                logger2 = get_logger('module.two.integ_test')
                
                # Log messages
                logger1.info('Message from module one')
                logger2.warning('Warning from module two')
                logger1.debug('Debug message - should not appear')  # Below INFO level
                
                # Flush handlers
                for handler in root_logger.handlers:
                    handler.flush()
                
                # Verify log file content
                with open(log_file, 'r') as f:
                    content = f.read()
                
                assert 'Message from module one' in content
                assert 'Warning from module two' in content
                assert 'module.one.integ_test' in content
                assert 'module.two.integ_test' in content
                assert 'Debug message' not in content  # Should be filtered out
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for handler in root_logger.handlers:
                    handler.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)


class TestSuppressLogging:
    """Tests for suppress_logging context manager."""

    def _save_and_clear_handlers(self):
        """Save current handlers and clear them."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        root_logger.handlers.clear()
        return root_logger, original_handlers, original_level

    def _restore_handlers(self, root_logger, original_handlers, original_level):
        """Restore original handlers."""
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers.clear()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)

    def test_suppress_logging_filters_messages(self):
        """Warnings are suppressed inside context and level restored after."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()

        try:
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.DEBUG)

            logger = get_logger('suppress.test')

            with suppress_logging(logging.ERROR):
                logger.warning('skip this warning')
                logger.error('keep this error')

            logger.warning('warning after restore')

            for h in root_logger.handlers:
                h.flush()

            content = stream.getvalue()

            assert 'skip this warning' not in content
            assert 'keep this error' in content
            assert 'warning after restore' in content
        finally:
            for h in root_logger.handlers:
                h.close()
            root_logger.handlers.clear()
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_suppress_logging_restores_on_exception(self):
        """Log level is restored even if an exception is raised inside context."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()

        try:
            root_logger.setLevel(logging.INFO)

            with pytest.raises(ValueError):
                with suppress_logging(logging.CRITICAL):
                    raise ValueError('boom')

            assert root_logger.level == logging.INFO
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_child_loggers_inherit_configuration(self):
        """Test that child loggers inherit root logger configuration."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            setup_logging(level=logging.WARNING)
            
            child_logger = get_logger('parent.child.inherit_test')
            
            # Child logger should respect root logger's level
            assert child_logger.getEffectiveLevel() == logging.WARNING
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)

    def test_multiple_file_loggers_in_sequence(self):
        """Test that setup_logging properly handles being called after reset."""
        root_logger, original_handlers, original_level = self._save_and_clear_handlers()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # First setup
                log_file1 = os.path.join(tmpdir, 'log1.log')
                setup_logging(log_file=log_file1)
                
                logger = get_logger('test.multiple.seq')
                logger.info('First message')
                
                # Flush
                for h in root_logger.handlers:
                    h.flush()
                
                with open(log_file1, 'r') as f:
                    assert 'First message' in f.read()
                
                # Close and reset handlers before setting up again
                for h in root_logger.handlers:
                    h.close()
                root_logger.handlers.clear()
                
                log_file2 = os.path.join(tmpdir, 'log2.log')
                setup_logging(log_file=log_file2)
                
                logger.info('Second message')
                
                for h in root_logger.handlers:
                    h.flush()
                
                with open(log_file2, 'r') as f:
                    content = f.read()
                    assert 'Second message' in content
                    assert 'First message' not in content
                
                # Close handlers before exiting temp directory context (Windows file locking)
                for handler in root_logger.handlers:
                    handler.close()
                root_logger.handlers.clear()
        finally:
            self._restore_handlers(root_logger, original_handlers, original_level)


class TestModuleLevelSetup:
    """Test the module-level auto-setup behavior."""

    def test_default_log_path_is_set(self):
        """Test that _default_log_path is properly configured."""
        from src.utils import logger as logger_module
        
        expected_path = os.path.normpath(os.path.join(os.path.expanduser('~'), '.seagent', 'logs', 'seagent.log'))
        assert logger_module._default_log_path == expected_path

    def test_module_auto_initializes_logging(self):
        """Test that logging is configured when module is imported."""
        # The module should have already called setup_logging
        # We just verify that root logger has handlers
        root_logger = logging.getLogger()
        
        # Should have at least one handler (might include pytest's handlers)
        assert len(root_logger.handlers) >= 1
