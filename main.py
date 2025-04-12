from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty

from PIL import Image, ImageDraw, ImageFilter



FILENAME = 'handwriting.jpg'


class Painter(Widget):
    pencil_size = NumericProperty(15)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = (800, 600)
        self.image = Image.new("RGB", self.canvas_size, "white")
        self.draw = ImageDraw.Draw(self.image)
        self.drawing = False
        self.last_pos = None

    def on_size(self, *args):
        self.canvas_size = (int(self.width), int(self.height))
        self._reset_image()

    def _reset_image(self):
        self.image = Image.new("RGB", self.canvas_size, "white")
        self.draw = ImageDraw.Draw(self.image)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.drawing = True
            self.last_pos = touch.pos

    def on_touch_move(self, touch):
        if self.drawing and self.last_pos:
            with self.canvas:
                from kivy.graphics import Color, Line
                Color(0, 0, 0)
                Line(points=[*self.last_pos, *touch.pos], width=self.pencil_size)
            self.draw.line([
                self.last_pos[0], self.height - self.last_pos[1],
                touch.pos[0], self.height - touch.pos[1]
            ], fill='black', width=int(self.pencil_size))
            self.last_pos = touch.pos

    def on_touch_up(self, touch):
        self.drawing = False
        self.last_pos = None

    def clear_canvas(self):
        self._reset_image()
        self.canvas.clear()

    def save_image(self):
        smoothed = self.image.filter(ImageFilter.SMOOTH_MORE)
        smoothed.save(FILENAME)
        popup = Popup(title="Saved",
                      content=Label(text=f"Saved as {FILENAME}"),
                      size_hint=(0.4, 0.3))
        popup.open()


class PainterApp(App):
    def build(self):
        from kivy.lang import Builder
        return Builder.load_file("painter_main.kv")


if __name__ == '__main__':
    PainterApp().run()
