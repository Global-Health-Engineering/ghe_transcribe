import logging
import os

import ipywidgets as widgets
from IPython.display import clear_output, display

# Import the core transcription function and config from your package
from ghe_transcribe.core import (
    ComputeTypeChoice,  # Enum for compute type choices
    DeviceChoice,  # Enum for device choices
    WhisperModelChoice,  # Enum for Whisper model choices
    transcribe_config,  # Default configuration
    transcribe_core,
)

# To get the parent directory:
root_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# Output directory for transcribed files
output_dir = os.path.join(root_path, "output")
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists


class GheTranscribeApp:
    def __init__(self):
        self.common_widget_layout = widgets.Layout(width="90%", margin="5px auto")
        self.common_widget_style = {"description_width": "150px"}
        self._setup_ui()
        self._set_initial_widget_states()
        self._observe_widget_changes()

    def _set_initial_widget_states(self):
        """Sets the initial disabled/hidden states for widgets."""
        self.advanced_widgets_box.layout.display = "none"

    def _create_dropdown_from_enum(
        self, enum_class, description, default_value_from_config
    ):
        """Helper to create consistent dropdown widgets from an Enum."""
        options = [(member.value, member.value) for member in enum_class]
        # Ensure the default value from config is valid for the enum
        # If default_value_from_config is None or not in enum, pick the first option
        default_selected_value = default_value_from_config
        if default_selected_value not in [opt[1] for opt in options]:
            default_selected_value = options[0][1]  # Fallback to first option

        return widgets.Dropdown(
            options=options,
            value=default_selected_value,
            description=description,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

    def _setup_ui(self):
        """Defines all the UI widgets and their initial layout."""
        # Basic Options
        self.audio_uploader = widgets.FileUpload(
            multiple=False,
            description="Upload Audio",
            accept=".wav,.mp3,.flac,.ogg",  # Specify accepted audio formats
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

        self.trim_input = widgets.FloatText(
            value=transcribe_config.get("trim") or 0.0,  # Default to 0.0 if None
            description="Trim (s):",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
            step=0.5,
        )

        # Speakers options for pyannote.audio
        speakers_options = [("Auto-detect", None)]
        for i in range(1, 11):  # From 1 to 10 speakers
            speakers_options.append((str(i), i))

        self.num_speakers_dropdown = widgets.Dropdown(
            options=speakers_options,
            value=transcribe_config.get("num_speakers"),
            description="Num. Speakers:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

        self.whisper_model_dropdown = self._create_dropdown_from_enum(
            WhisperModelChoice, "Whisper Model:", transcribe_config.get("whisper_model")
        )

        self.advanced_options_checkbox = widgets.Checkbox(
            value=False,
            description="Advanced Options",
            indent=False,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

        self.basic_widgets_box = widgets.VBox(
            [
                self.audio_uploader,
                self.trim_input,
                self.num_speakers_dropdown,
                self.whisper_model_dropdown,
                self.advanced_options_checkbox,
            ],
            layout=widgets.Layout(
                width="50%", margin="0 auto", border="1px solid #ccc", padding="15px"
            ),
        )

        # Advanced Options (Faster Whisper & VAD parameters)
        self.device_dropdown = self._create_dropdown_from_enum(
            DeviceChoice, "Device:", transcribe_config.get("device")
        )
        self.cpu_threads_input = widgets.IntText(
            value=transcribe_config.get("cpu_threads")
            or 0,  # 0 can mean auto/system default
            description="CPU Threads:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.min_speakers_input = widgets.IntText(
            value=transcribe_config.get("min_speakers") or 1,
            description="Min Speakers:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.max_speakers_input = widgets.IntText(
            value=transcribe_config.get("max_speakers") or 10,
            description="Max Speakers:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.device_index_input = widgets.IntText(
            value=transcribe_config.get("device_index") or 0,
            description="Device Index:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.compute_type_dropdown = self._create_dropdown_from_enum(
            ComputeTypeChoice, "Compute Type:", transcribe_config.get("compute_type")
        )
        self.beam_size_input = widgets.IntText(
            value=transcribe_config.get("beam_size"),
            description="Beam Size:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.temperature_input = widgets.FloatText(
            value=transcribe_config.get("temperature"),
            description="Temperature:",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
            step=0.1,
        )
        self.word_timestamps_checkbox = widgets.Checkbox(
            value=transcribe_config.get("word_timestamps") or False,
            description="Word Timestamps",
            indent=False,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.vad_filter_checkbox = widgets.Checkbox(
            value=transcribe_config.get("vad_filter") or False,
            description="VAD Filter",
            indent=False,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.min_silence_duration_ms_input = widgets.IntText(
            value=transcribe_config.get("min_silence_duration_ms"),
            description="Min Silence (ms):",
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

        self.save_output_checkbox = widgets.Checkbox(
            value=transcribe_config.get("save_output") or True,
            description="Save Output (.txt, .srt)",
            indent=False,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )
        self.info_checkbox = widgets.Checkbox(
            value=transcribe_config.get("info") or True,
            description="Print Language Info",
            indent=False,
            layout=self.common_widget_layout,
            style=self.common_widget_style,
        )

        self.advanced_widgets_box = widgets.VBox(
            [
                self.min_speakers_input,
                self.max_speakers_input,
                self.device_dropdown,
                self.cpu_threads_input,
                self.device_index_input,
                self.compute_type_dropdown,
                self.beam_size_input,
                self.temperature_input,
                self.word_timestamps_checkbox,
                self.vad_filter_checkbox,
                self.min_silence_duration_ms_input,
                self.save_output_checkbox,
                self.info_checkbox,
            ],
            layout=widgets.Layout(
                width="50%", margin="0 auto", border="1px solid #ccc", padding="15px"
            ),
        )

        # Run Button
        self.run_button = widgets.Button(
            description="Run Transcription",
            layout=widgets.Layout(width="200px", margin="10px auto"),
            button_style="primary",
        )

        # Run button box
        self.run_widgets_box = widgets.VBox(
            [
                self.run_button,
            ],
            layout=widgets.Layout(
                width="50%",
                margin="0 auto",
                border="1px solid #ccc",
                padding="15px",
                display="flex",
                flex_flow="column",
                align_items="center",
            ),
        )

        # Output area with simple styling
        self.output_area = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ddd",
                padding="10px"
            )
        )

        # Output container box
        self.output_widgets_box = widgets.VBox(
            [
                self.output_area,
            ],
            layout=widgets.Layout(
                width="90%",
                margin="10px auto",
                border="1px solid #ccc",
                padding="15px",
                display="flex",
                flex_flow="column",
            ),
        )

    def _observe_widget_changes(self):
        """Sets up observers for widget value changes."""
        self.advanced_options_checkbox.observe(
            self._on_advanced_options_checkbox_change, names="value"
        )
        self.run_button.on_click(self._on_run_button_click)

    def _on_advanced_options_checkbox_change(self, change):
        """Callback for advanced options checkbox."""
        self.advanced_widgets_box.layout.display = "block" if change.new else "none"

    def _on_run_button_click(self, b):
        """Callback for the run button, initiating transcription."""
        # Use output widget for controlled display
        with self.output_area:
            clear_output(wait=True)
            logger.info("Starting transcription...")
            print("Starting transcription...")

            if not self.audio_uploader.value:
                logger.warning("No audio file uploaded")
                print("Please upload an audio file first.")
                return

            try:
                # Handle uploaded file
                file_metadata = self.audio_uploader.value[0]
                uploaded_file_name = file_metadata["name"]
                uploaded_content_bytes = file_metadata["content"].tobytes()

                # Ensure 'media' directory exists within the root_path
                media_dir = os.path.join(root_path, "media")
                os.makedirs(media_dir, exist_ok=True)

                audio_file_path = os.path.join(media_dir, uploaded_file_name)
                with open(audio_file_path, "wb") as f:
                    f.write(uploaded_content_bytes)
                logger.info(f"Uploaded audio saved to: {audio_file_path}")
                print(f"Uploaded audio saved to: {audio_file_path}")

                # Prepare arguments for transcribe
                kwargs = {
                    "file": audio_file_path,
                    "trim": self.trim_input.value if self.trim_input.value > 0 else None,
                    "device": self.device_dropdown.value,
                    "cpu_threads": self.cpu_threads_input.value
                    if self.cpu_threads_input.value > 0
                    else None,
                    "whisper_model": self.whisper_model_dropdown.value,
                    "device_index": self.device_index_input.value,
                    "compute_type": self.compute_type_dropdown.value,
                    "beam_size": self.beam_size_input.value,
                    "temperature": self.temperature_input.value,
                    "word_timestamps": self.word_timestamps_checkbox.value,
                    "vad_filter": self.vad_filter_checkbox.value,
                    "min_silence_duration_ms": self.min_silence_duration_ms_input.value,
                    "save_output": self.save_output_checkbox.value,
                    "info": self.info_checkbox.value,
                }

                # Diarization specific arguments
                if self.num_speakers_dropdown.value is not None:
                    kwargs["num_speakers"] = self.num_speakers_dropdown.value
                else:
                    # If "Auto-detect" is selected for num_speakers, use min/max
                    kwargs["min_speakers"] = self.min_speakers_input.value
                    kwargs["max_speakers"] = self.max_speakers_input.value

                # Call the ghe_transcribe function
                transcribe_core(**kwargs)

            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                print(f"An unexpected error occurred: {e}")
                import traceback

                traceback.print_exc()  # Print full traceback for debugging

    def display_app(self):
        """Displays all the UI components."""
        display(self.basic_widgets_box, self.advanced_widgets_box, self.run_widgets_box, self.output_widgets_box)


def execute():
    """Execute the transcription app with auto-installation if needed."""
    import subprocess
    import sys

    try:
        # Check if we can create the app (tests if dependencies are available)
        app = GheTranscribeApp()
        app.display_app()
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Installing package in development mode...")

        try:
            # Install the package in development mode
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            print("Installation successful! Please restart your kernel and run again.")
            print("In Jupyter: Kernel → Restart Kernel, then re-run your cell.")
        except subprocess.CalledProcessError as install_error:
            print(f"Installation failed: {install_error}")
            print('Please install manually: pip install -e ".[ui]"')
            raise


if __name__ == "__main__":
    execute()
