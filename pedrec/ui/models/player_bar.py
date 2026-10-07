from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QStyle, QToolButton, QWidget, QStyleOptionSlider

from pedrec.ui import theme

SKIP_SECONDS = 5.0


class SeekSlider(QSlider):
    """Slider that jumps to the clicked position; ``seek_requested`` is emitted on click / release."""
    seek_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sliderReleased.connect(lambda: self.seek_requested.emit(self.value()))

    def mousePressEvent(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                             QStyle.SubControl.SC_SliderHandle, self)
        position = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if not handle.contains(position):
            value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), position.x(), self.width())
            self.setValue(value)
            self.seek_requested.emit(value)
        super().mousePressEvent(event)


def _time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


class PlayerBar(QWidget):
    """Player controls: replay, skip back / forward, frame by frame, play / pause and a seek bar."""

    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.playing = True
        self.frame_count = 0
        style = self.style()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        def button(icon, tooltip, slot):
            b = QToolButton(self)
            b.setIcon(style.standardIcon(icon))
            b.setToolTip(tooltip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            layout.addWidget(b)
            return b

        self.replay_button = button(QStyle.StandardPixmap.SP_MediaSkipBackward, "Replay from the start (Home)",
                                    worker.replay)
        self.back_button = button(QStyle.StandardPixmap.SP_MediaSeekBackward,
                                  f"Back {SKIP_SECONDS:g} s (Left while playing)",
                                  lambda: worker.skip_seconds(-SKIP_SECONDS))
        self.prev_button = button(QStyle.StandardPixmap.SP_ArrowLeft, "Previous frame (Left while paused)",
                                  lambda: worker.step(-1))
        self.play_button = button(QStyle.StandardPixmap.SP_MediaPause, "Play / pause (Space)", worker.toggle_play)
        self.next_button = button(QStyle.StandardPixmap.SP_ArrowRight, "Next frame (Right while paused)",
                                  lambda: worker.step(1))
        self.forward_button = button(QStyle.StandardPixmap.SP_MediaSeekForward,
                                     f"Forward {SKIP_SECONDS:g} s (Right while playing)",
                                     lambda: worker.skip_seconds(SKIP_SECONDS))
        self.slider = SeekSlider(self)
        self.slider.seek_requested.connect(worker.seek)
        self.slider.sliderMoved.connect(self._show_time)
        layout.addWidget(self.slider, 1)
        self.time_label = QLabel("--:--", self)
        self.time_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; padding-left: 6px;")
        layout.addWidget(self.time_label)

        seekable = worker.seekable
        for widget in (self.replay_button, self.back_button, self.prev_button, self.next_button,
                       self.forward_button, self.slider):
            widget.setEnabled(seekable)
        worker.position_changed.connect(self.on_position)
        worker.playing_changed.connect(self.on_playing)

    # keyboard (shortcuts of the main window)
    def left(self):
        self.worker.step(-1) if not self.playing else self.worker.skip_seconds(-SKIP_SECONDS)

    def right(self):
        self.worker.step(1) if not self.playing else self.worker.skip_seconds(SKIP_SECONDS)

    def on_playing(self, playing: bool):
        self.playing = playing
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))

    def on_position(self, index: int, count: int):
        self.frame_count = count
        if count > 0:
            self.slider.setRange(0, count - 1)
            if not self.slider.isSliderDown():
                self.slider.setValue(index)
        self._show_time(index)

    def _show_time(self, index: int):
        fps = getattr(self.worker.source, "fps", 30.0) or 30.0
        if self.frame_count > 0:
            self.time_label.setText(f"{_time(index / fps)} / {_time((self.frame_count - 1) / fps)}   "
                                    f"frame {index + 1}/{self.frame_count}")
        else:
            self.time_label.setText(f"live   frame {index + 1}")
