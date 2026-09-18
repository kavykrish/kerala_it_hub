import json
import os
import re
import threading

import requests

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


# ============================================================
# BACKEND SERVER
# ============================================================
# This is the default backend URL baked into the APK. Replace it with
# your deployed backend's HTTPS URL (e.g. from Render/Railway) before
# building the release APK. Students can also change it later at
# runtime from the in-app Settings screen (gear icon) without needing
# a new APK, which is handy if you move hosts or are testing locally.

DEFAULT_API_BASE_URL = "https://YOUR-BACKEND-URL.onrender.com"

# Keep this in sync with the API_KEY environment variable on the
# backend (backend/main.py). Leave both blank while the backend has
# no API_KEY configured.
DEFAULT_API_KEY = ""

# Kivy's software-keyboard handling: slide the view up so the input
# field stays visible above the on-screen keyboard on Android.
Window.softinput_mode = "below_target"


def get_settings_path():

    return os.path.join(
        App.get_running_app().user_data_dir,
        "settings.json"
    )


def load_settings():

    path = get_settings_path()

    if os.path.exists(path):

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "api_base_url": data.get(
                    "api_base_url",
                    DEFAULT_API_BASE_URL
                ),
                "api_key": data.get(
                    "api_key",
                    DEFAULT_API_KEY
                )
            }

        except (json.JSONDecodeError, OSError):
            pass

    return {
        "api_base_url": DEFAULT_API_BASE_URL,
        "api_key": DEFAULT_API_KEY
    }


def save_settings(api_base_url, api_key):

    path = get_settings_path()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "api_base_url": api_base_url.strip(),
                "api_key": api_key.strip()
            },
            f
        )


# ============================================================
# COURSE CARD
# ============================================================

class CourseCard(BoxLayout):

    def __init__(self, text, **kwargs):

        super().__init__(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(5),
            size_hint_y=None,
            **kwargs
        )

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        with self.canvas.before:

            Color(
                0.08,
                0.08,
                0.08,
                1
            )

            self.background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(12)]
            )

            Color(
                0.25,
                0.25,
                0.25,
                1
            )

            self.border = Line(
                rounded_rectangle=(
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    dp(12)
                ),
                width=1
            )

        self.bind(
            pos=self.update_background,
            size=self.update_background
        )

        # ----------------------------------------------------
        # Course text
        # ----------------------------------------------------

        self.label = Label(
            text=text,
            font_size=dp(15),
            halign="left",
            valign="top",
            size_hint_y=None,
            size_hint_x=1,
            padding=(dp(5), dp(5))
        )

        self.label.bind(
            width=self.update_text_width
        )

        self.label.bind(
            texture_size=self.update_card_height
        )

        self.add_widget(
            self.label
        )

    # ========================================================
    # UPDATE BACKGROUND
    # ========================================================

    def update_background(
        self,
        *args
    ):

        self.background.pos = self.pos
        self.background.size = self.size

        self.border.rounded_rectangle = (
            self.x,
            self.y,
            self.width,
            self.height,
            dp(12)
        )

    # ========================================================
    # TEXT WIDTH
    # ========================================================

    def update_text_width(
        self,
        instance,
        width
    ):

        instance.text_size = (
            width - dp(10),
            None
        )

    # ========================================================
    # CARD HEIGHT
    # ========================================================

    def update_card_height(
        self,
        instance,
        texture_size
    ):

        self.height = (
            texture_size[1]
            + dp(35)
        )


# ============================================================
# KERALA IT HUB APP
# ============================================================

class KeralaITHubApp(App):

    # ========================================================
    # BUILD
    # ========================================================

    def build(self):

        # ----------------------------------------------------
        # Load persisted settings (backend URL / API key)
        # ----------------------------------------------------

        settings = load_settings()

        self.api_base_url = settings["api_base_url"]
        self.api_key = settings["api_key"]

        # ----------------------------------------------------
        # Main layout
        # ----------------------------------------------------

        main_layout = BoxLayout(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(10)
        )

        # ====================================================
        # TOP BAR (TITLE + SETTINGS)
        # ====================================================

        top_bar = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(55)
        )

        title = Label(
            text="KERALA IT HUB",
            font_size=dp(30),
            bold=True,
            halign="left",
            valign="middle"
        )

        title.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        settings_button = Button(
            text="Settings",
            font_size=dp(13),
            size_hint_x=None,
            width=dp(90)
        )

        settings_button.bind(
            on_press=self.open_settings
        )

        top_bar.add_widget(title)
        top_bar.add_widget(settings_button)

        # ====================================================
        # SUBTITLE
        # ====================================================

        subtitle = Label(
            text=(
                "AI-Powered IT Course & "
                "Institute Navigator"
            ),
            font_size=dp(16),
            size_hint_y=None,
            height=dp(35)
        )

        # ====================================================
        # QUESTION LABEL
        # ====================================================

        question_label = Label(
            text="Ask your question:",
            font_size=dp(17),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(30)
        )

        question_label.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        # ====================================================
        # QUESTION INPUT
        # ====================================================

        self.question_input = TextInput(
            hint_text=(
                "Example: Which Python courses "
                "are available in Kerala?"
            ),
            multiline=True,
            font_size=dp(16),
            size_hint_y=None,
            height=dp(75),
            padding=dp(10)
        )

        # ====================================================
        # ASK BUTTON
        # ====================================================

        self.ask_button = Button(
            text="ASK",
            font_size=dp(18),
            bold=True,
            size_hint_y=None,
            height=dp(48)
        )

        self.ask_button.bind(
            on_press=self.ask_question
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = Label(
            text="",
            font_size=dp(14),
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=dp(28)
        )

        # ====================================================
        # ANSWER TITLE
        # ====================================================

        answer_title = Label(
            text="Answer",
            font_size=dp(20),
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(32)
        )

        answer_title.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        # ====================================================
        # ANSWER SCROLL
        # ====================================================

        self.answer_scroll = ScrollView(
            size_hint_y=1,
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(7)
        )

        # ====================================================
        # ANSWER LAYOUT
        # ====================================================

        self.answer_layout = BoxLayout(
            orientation="vertical",
            spacing=dp(12),
            padding=(dp(5), dp(5)),
            size_hint_y=None
        )

        self.answer_layout.bind(
            minimum_height=
            self.answer_layout.setter(
                "height"
            )
        )

        self.answer_scroll.add_widget(
            self.answer_layout
        )

        # ====================================================
        # SOURCES TITLE
        # ====================================================

        sources_title = Label(
            text="Sources",
            font_size=dp(20),
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(32)
        )

        sources_title.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        # ====================================================
        # SOURCES SCROLL
        # ====================================================

        self.sources_scroll = ScrollView(
            size_hint_y=None,
            height=dp(105),
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(7)
        )

        self.sources_label = Label(
            text="Sources will appear here.",
            font_size=dp(13),
            halign="left",
            valign="top",
            size_hint_y=None,
            size_hint_x=1,
            padding=(dp(10), dp(10))
        )

        self.sources_label.bind(
            width=self.update_sources_width
        )

        self.sources_label.bind(
            texture_size=self.update_sources_height
        )

        self.sources_scroll.add_widget(
            self.sources_label
        )

        # ====================================================
        # ADD WIDGETS
        # ====================================================

        main_layout.add_widget(top_bar)
        main_layout.add_widget(subtitle)
        main_layout.add_widget(question_label)
        main_layout.add_widget(self.question_input)
        main_layout.add_widget(self.ask_button)
        main_layout.add_widget(self.status_label)
        main_layout.add_widget(answer_title)
        main_layout.add_widget(self.answer_scroll)
        main_layout.add_widget(sources_title)
        main_layout.add_widget(self.sources_scroll)

        return main_layout

    # ========================================================
    # OPEN SETTINGS
    # ========================================================

    def open_settings(
        self,
        instance
    ):

        form = BoxLayout(
            orientation="vertical",
            spacing=dp(10),
            padding=dp(15)
        )

        form.add_widget(
            Label(
                text="Backend server URL",
                font_size=dp(14),
                size_hint_y=None,
                height=dp(22),
                halign="left"
            )
        )

        url_input = TextInput(
            text=self.api_base_url,
            multiline=False,
            font_size=dp(14),
            size_hint_y=None,
            height=dp(42)
        )

        form.add_widget(url_input)

        form.add_widget(
            Label(
                text="API key (leave blank if none)",
                font_size=dp(14),
                size_hint_y=None,
                height=dp(22),
                halign="left"
            )
        )

        key_input = TextInput(
            text=self.api_key,
            multiline=False,
            password=True,
            font_size=dp(14),
            size_hint_y=None,
            height=dp(42)
        )

        form.add_widget(key_input)

        button_row = BoxLayout(
            orientation="horizontal",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(45)
        )

        save_button = Button(text="Save")
        cancel_button = Button(text="Cancel")

        button_row.add_widget(cancel_button)
        button_row.add_widget(save_button)

        form.add_widget(button_row)

        popup = Popup(
            title="Settings",
            content=form,
            size_hint=(0.9, 0.5)
        )

        def on_save(_instance):

            new_url = url_input.text.strip().rstrip("/")
            new_key = key_input.text.strip()

            if not new_url:
                return

            self.api_base_url = new_url
            self.api_key = new_key

            save_settings(new_url, new_key)

            popup.dismiss()

        save_button.bind(on_press=on_save)
        cancel_button.bind(on_press=popup.dismiss)

        popup.open()

    # ========================================================
    # SOURCE WIDTH
    # ========================================================

    def update_sources_width(
        self,
        instance,
        width
    ):

        instance.text_size = (
            width - dp(20),
            None
        )

    # ========================================================
    # SOURCE HEIGHT
    # ========================================================

    def update_sources_height(
        self,
        instance,
        texture_size
    ):

        instance.height = max(
            texture_size[1] + dp(20),
            dp(70)
        )

    # ========================================================
    # CLEAN MARKDOWN
    # ========================================================

    def clean_markdown(
        self,
        text
    ):

        if not text:
            return ""

        text = text.replace(
            "**",
            ""
        )

        text = text.replace(
            "__",
            ""
        )

        text = text.replace(
            "<br>",
            "\n"
        )

        text = text.replace(
            "<br/>",
            "\n"
        )

        text = text.replace(
            "<br />",
            "\n"
        )

        return text.strip()

    # ========================================================
    # NORMAL TEXT LABEL
    # ========================================================

    def create_text_label(
        self,
        text,
        font_size=15,
        bold=False
    ):

        label = Label(
            text=text,
            font_size=dp(font_size),
            bold=bold,
            halign="left",
            valign="top",
            size_hint_y=None,
            size_hint_x=1,
            padding=(dp(10), dp(8))
        )

        label.bind(
            width=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value - dp(20), None)
            )
        )

        label.bind(
            texture_size=lambda instance, value:
            setattr(
                instance,
                "height",
                value[1] + dp(16)
            )
        )

        return label

    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    def display_answer(
        self,
        answer
    ):

        self.answer_layout.clear_widgets()

        if not answer:

            self.answer_layout.add_widget(
                self.create_text_label(
                    "No answer available."
                )
            )

            return

        answer = self.clean_markdown(
            answer
        )

        lines = answer.splitlines()

        table_rows = []
        normal_lines = []

        # ----------------------------------------------------
        # Process answer
        # ----------------------------------------------------

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Detect markdown table
            # ------------------------------------------------

            if "|" in line:

                parts = [
                    item.strip()
                    for item in
                    line.strip("|").split("|")
                ]

                parts = [
                    item
                    for item in parts
                    if item
                ]

                # Ignore separator
                if all(
                    re.fullmatch(
                        r"[-:]+",
                        item.replace(" ", "")
                    )
                    for item in parts
                ):
                    continue

                if len(parts) >= 2:

                    table_rows.append(
                        parts
                    )

                    continue

            # ------------------------------------------------
            # Normal text
            # ------------------------------------------------

            normal_lines.append(
                line
            )

        # ====================================================
        # DISPLAY INTRODUCTION
        # ====================================================

        if normal_lines:

            intro_text = []

            for line in normal_lines:

                # Avoid displaying source section
                if line.lower() == "sources":
                    continue

                intro_text.append(line)

            if intro_text:

                self.answer_layout.add_widget(
                    self.create_text_label(
                        "\n\n".join(intro_text),
                        font_size=15
                    )
                )

        # ====================================================
        # DISPLAY COURSE CARDS
        # ====================================================

        if table_rows:

            # ------------------------------------------------
            # Section heading
            # ------------------------------------------------

            heading = self.create_text_label(
                "Courses found:",
                font_size=18,
                bold=True
            )

            self.answer_layout.add_widget(
                heading
            )

            header = table_rows[0]

            # ------------------------------------------------
            # Each row becomes a card
            # ------------------------------------------------

            for row in table_rows[1:]:

                if not row:
                    continue

                card_lines = []

                for index, value in enumerate(row):

                    if index >= len(header):
                        continue

                    field = header[index]

                    if not value:
                        continue

                    # ------------------------------------------------
                    # Make field name cleaner
                    # ------------------------------------------------

                    field = field.strip()

                    card_lines.append(
                        f"{field}: {value}"
                    )

                if card_lines:

                    card_text = "\n".join(
                        card_lines
                    )

                    card = CourseCard(
                        card_text
                    )

                    self.answer_layout.add_widget(
                        card
                    )

        # ====================================================
        # FALLBACK
        # ====================================================

        if not table_rows and normal_lines:

            # Already displayed above
            pass

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(
        self,
        instance
    ):

        question = (
            self.question_input
            .text
            .strip()
        )

        if not question:

            self.status_label.text = (
                "Please enter a question."
            )

            return

        if (
            not self.api_base_url
            or "YOUR-BACKEND-URL" in self.api_base_url
        ):

            self.status_label.text = (
                "Please set the backend server URL "
                "in Settings first."
            )

            return

        # ----------------------------------------------------
        # Disable button
        # ----------------------------------------------------

        self.ask_button.disabled = True

        self.ask_button.text = (
            "SEARCHING..."
        )

        self.status_label.text = (
            "Searching Kerala IT course information..."
        )

        # ----------------------------------------------------
        # Loading
        # ----------------------------------------------------

        self.answer_layout.clear_widgets()

        self.answer_layout.add_widget(
            self.create_text_label(
                (
                    "Searching the web and retrieving "
                    "relevant course information..."
                ),
                font_size=15
            )
        )

        self.sources_label.text = (
            "Retrieving sources..."
        )

        # ----------------------------------------------------
        # Thread
        # ----------------------------------------------------

        thread = threading.Thread(
            target=self.send_question_to_api,
            args=(question,),
            daemon=True
        )

        thread.start()

    # ========================================================
    # API REQUEST
    # ========================================================

    def send_question_to_api(
        self,
        question
    ):

        try:

            headers = {}

            if self.api_key:
                headers["X-API-Key"] = self.api_key

            response = requests.post(
                f"{self.api_base_url}/ask",
                json={
                    "question": question
                },
                headers=headers,
                timeout=180
            )

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            if response.status_code != 200:

                self.update_ui(
                    "error",
                    (
                        "FastAPI returned "
                        f"status code "
                        f"{response.status_code}."
                    ),
                    []
                )

                return

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            data = response.json()

            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            if data.get("status") != "success":

                self.update_ui(
                    "error",
                    data.get(
                        "message",
                        "Unknown API error."
                    ),
                    []
                )

                return

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            answer = data.get(
                "answer",
                "No answer returned."
            )

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            sources = data.get(
                "sources",
                []
            )

            # ------------------------------------------------
            # Update UI
            # ------------------------------------------------

            self.update_ui(
                "success",
                answer,
                sources
            )

        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            self.update_ui(
                "error",
                (
                    "Could not connect to FastAPI.\n\n"
                    "Please make sure the server is running."
                ),
                []
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            self.update_ui(
                "error",
                (
                    "The request took too long.\n\n"
                    "Please try again."
                ),
                []
            )

        # ----------------------------------------------------
        # Other error
        # ----------------------------------------------------

        except Exception as e:

            self.update_ui(
                "error",
                f"Error: {str(e)}",
                []
            )

    # ========================================================
    # UPDATE UI
    # ========================================================

    def update_ui(
        self,
        status,
        answer,
        sources
    ):

        Clock.schedule_once(
            lambda dt:
            self.finish_ui_update(
                status,
                answer,
                sources
            )
        )

    # ========================================================
    # FINISH UI
    # ========================================================

    def finish_ui_update(
        self,
        status,
        answer,
        sources
    ):

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if status == "success":

            self.status_label.text = (
                "Answer generated successfully."
            )

        else:

            self.status_label.text = (
                "Request failed."
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        if status == "success":

            self.display_answer(
                answer
            )

        else:

            self.answer_layout.clear_widgets()

            self.answer_layout.add_widget(
                self.create_text_label(
                    answer,
                    font_size=15
                )
            )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        source_text = ""

        for index, source in enumerate(
            sources,
            start=1
        ):

            title = source.get(
                "title",
                "Unknown source"
            )

            url = source.get(
                "url",
                ""
            )

            source_text += (
                f"{index}. {title}\n"
                f"{url}\n\n"
            )

        if source_text:

            self.sources_label.text = (
                source_text.strip()
            )

        else:

            self.sources_label.text = (
                "No sources available."
            )

        # ----------------------------------------------------
        # Reset scroll
        # ----------------------------------------------------

        Clock.schedule_once(
            self.reset_scroll,
            0.5
        )

        # ----------------------------------------------------
        # Enable button
        # ----------------------------------------------------

        self.ask_button.disabled = False

        self.ask_button.text = "ASK"

    # ========================================================
    # RESET SCROLL
    # ========================================================

    def reset_scroll(
        self,
        dt
    ):

        self.answer_scroll.scroll_y = 1

        self.sources_scroll.scroll_y = 1


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    KeralaITHubApp().run()