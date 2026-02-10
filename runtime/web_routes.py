import asyncio
import logging
import os

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from agents.definitions import AGENTS_CONFIG
from runtime.runtime_context import RuntimeContext

logger = logging.getLogger("LotusSystem")


def create_web_blueprint(app, play_voice_func):
    web_bp = Blueprint("web", __name__)

    @web_bp.route("/")
    def index():
        return render_template("index.html")

    @web_bp.route("/api/toggle_voice", methods=["POST"])
    def toggle_voice_api():
        data = request.json
        if data and "active" in data:
            RuntimeContext.voice_mode_active = data["active"]
        else:
            RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active

        status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
        logger.info(f"🎙️ Mikrofon Modu Değiştirildi: {status_msg}")
        return jsonify({"status": "success", "active": RuntimeContext.voice_mode_active})

    @web_bp.route("/api/chat_history", methods=["GET"])
    def get_chat_history():
        agent_name = request.args.get("agent", "ATLAS")
        if RuntimeContext.engine and RuntimeContext.engine.memory:
            try:
                history = RuntimeContext.engine.memory.get_agent_history_for_web(agent_name, limit=20)
                return jsonify({"status": "success", "history": history})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
        return jsonify({"status": "error", "message": "Hafıza modülü hazır değil."})

    @web_bp.route("/api/chat", methods=["POST"])
    def web_chat():
        user_msg = request.form.get("message", "")
        target_agent_req = request.form.get("target_agent", "GENEL")

        uploaded_file = request.files.get("file")
        auth_file = request.files.get("auth_frame")
        file_path = None

        if uploaded_file and uploaded_file.filename != "":
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(file_path)

        if not user_msg and not file_path:
            return jsonify({"status": "error", "reply": "Mesaj içeriği boş."})

        if user_msg.lower().strip() in ["hafızayı sil", "hafızayı temizle"]:
            if RuntimeContext.engine and RuntimeContext.engine.memory:
                RuntimeContext.engine.memory.clear_history()
                return jsonify({"status": "success", "agent": "SİSTEM", "reply": "Hafıza başarıyla temizlendi."})

        identified_user = None
        frame_present = False

        if auth_file and RuntimeContext.security_instance:
            try:
                file_bytes = np.frombuffer(auth_file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_present = True
                    identified_user = RuntimeContext.security_instance.check_static_frame(frame)
            except Exception as e:
                logger.error(f"Web Auth İşleme Hatası: {e}")

        if identified_user:
            sec_result = ("ONAYLI", identified_user, None)
        elif frame_present:
            sec_result = ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
        else:
            sec_result = ("SORGULAMA", {"name": "Web Kullanıcısı", "level": 1}, "KAMERA_YOK")

        try:
            group_triggers = ["millet", "ekip", "herkes", "gençler", "arkadaşlar", "team", "tüm ekip", "hepiniz"]
            is_group_call = target_agent_req == "GENEL" and any(t in user_msg.lower() for t in group_triggers)

            if is_group_call and RuntimeContext.engine and RuntimeContext.loop:
                future = asyncio.run_coroutine_threadsafe(
                    RuntimeContext.engine.get_team_response(user_msg, sec_result),
                    RuntimeContext.loop,
                )
                replies_list = future.result(timeout=120)
                return jsonify({"status": "success", "replies": replies_list})

            final_agent = RuntimeContext.active_web_agent
            if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
                final_agent = target_agent_req
            else:
                if RuntimeContext.engine:
                    detected_agent = RuntimeContext.engine.determine_agent(user_msg)
                    if detected_agent:
                        final_agent = detected_agent
                    else:
                        final_agent = "ATLAS"

            RuntimeContext.active_web_agent = final_agent

            if RuntimeContext.loop and RuntimeContext.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    RuntimeContext.engine.get_response(final_agent, user_msg, sec_result, file_path=file_path),
                    RuntimeContext.loop,
                )
                response_data = future.result(timeout=90)

                if RuntimeContext.voice_mode_active:
                    RuntimeContext.executor.submit(
                        play_voice_func,
                        response_data["content"],
                        response_data["agent"],
                        RuntimeContext.state_manager,
                    )

                return jsonify(
                    {
                        "status": "success",
                        "agent": response_data["agent"],
                        "reply": response_data["content"],
                    }
                )
            return jsonify({"status": "error", "reply": "Lotus motoru şu an hazır değil."})

        except Exception as e:
            logger.error(f"Web Chat İşlem Hatası: {e}")
            return jsonify({"status": "error", "reply": f"Sistem hatası oluştu: {str(e)}"})

    @web_bp.route("/webhook", methods=["GET", "POST"])
    def webhook_handler():
        if request.method == "GET":
            verify_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "lotus_ai_guvenlik_tokeni")
            mode = request.args.get("hub.mode")
            token = request.args.get("hub.verify_token")
            challenge = request.args.get("hub.challenge")
            if mode == "subscribe" and token == verify_token:
                return challenge, 200
            return "Verification failed", 403

        if request.method == "POST":
            try:
                data = request.json
                parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
                if parsed:
                    RuntimeContext.msg_queue.put(parsed)
                    return jsonify({"status": "ok"}), 200
                return jsonify({"status": "ignored"}), 200
            except Exception as e:
                logger.error(f"Webhook Mesaj Hatası: {e}")
                return jsonify({"status": "error"}), 500

    return web_bp


def run_flask(app):
    try:
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Flask Sunucu Hatası: {e}")
