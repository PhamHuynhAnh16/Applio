import gradio as gr
from core import run_onnx_export_script

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def model_exporter_tab():
    with gr.Column():
        model_name = gr.Textbox(
            label=i18n("Path to Model"),
            info=i18n("Introduce the model pth path"),
            placeholder=i18n("Introduce the model pth path"),
            interactive=True,
        )
        model_output = gr.File(
            label=i18n("Output Model"),
            value=None,
            interactive=False,
        )
        model_exporter_button = gr.Button(i18n("Export"))
        model_exporter_button.click(
            fn=run_onnx_export_script,
            inputs=[model_name],
            outputs=[model_output],
        )
