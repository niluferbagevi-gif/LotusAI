"""
LotusAI Flask Server
Sürüm: 2.5.3
Açıklama: Web dashboard backend

Endpoints:
- / : Ana sayfa
- /api/chat : Mesaj gönderme
- /api/chat_history : Geçmiş sohbet
- /api/toggle_voice : Ses modu
- /webhook : Meta webhook
"""

import os
import logging
import asyncio
import cv2
import numpy as np
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
from config import Config
from core.runtime import RuntimeContext
from core.audio import play_voice
from agents.definitions import AGENTS_CONFIG

logger = logging.getLogger("LotusServer")


# ═══════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════
app = Flask(
    __name__,
    template_folder=str(Config.TEMPLATE_DIR),
    static_folder=str(Config.STATIC_DIR)
)

app.config['UPLOAD_FOLDER'] = str(Config.UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# ═══════════════════════════════════════════════════════════════
# ROUTES - PAGES
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index() -> str:
    """
    Ana sayfa
    
    Returns:
        Rendered HTML
    """
    return render_template('index.html')


# ═══════════════════════════════════════════════════════════════
# ROUTES - API
# ═══════════════════════════════════════════════════════════════
@app.route('/api/toggle_voice', methods=['POST'])
def toggle_voice_api() -> Response:
    """
    Ses modu toggle
    
    Returns:
        JSON response
    """
    try:
        data = request.json
        
        if data and 'active' in data:
            RuntimeContext.voice_mode_active = data['active']
        else:
            RuntimeContext.voice_mode_active = not RuntimeContext.voice_mode_active
        
        status_msg = "AÇIK" if RuntimeContext.voice_mode_active else "KAPALI"
        logger.info(f"🎙️ Mikrofon modu: {status_msg}")
        
        return jsonify({
            "status": "success",
            "active": RuntimeContext.voice_mode_active
        })
    
    except Exception as e:
        logger.error(f"Voice toggle hatası: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/chat_history', methods=['GET'])
def get_chat_history() -> Response:
    """
    Sohbet geçmişi
    
    Query params:
        agent: Agent adı (default: ATLAS)
    
    Returns:
        JSON response
    """
    try:
        agent_name = request.args.get('agent', 'ATLAS')
        
        if not RuntimeContext.engine or not RuntimeContext.engine.memory:
            return jsonify({
                "status": "error",
                "message": "Hafıza modülü hazır değil"
            }), 503
        
        history = RuntimeContext.engine.memory.get_agent_history_for_web(
            agent_name,
            limit=20
        )
        
        return jsonify({
            "status": "success",
            "history": history
        })
    
    except Exception as e:
        logger.error(f"Chat history hatası: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def web_chat() -> Response:
    """
    Web chat endpoint
    
    Form data:
        message: Mesaj metni
        target_agent: Hedef agent (optional)
    
    Files:
        file: Dosya upload (optional)
        auth_frame: Yüz tanıma frame (optional)
    
    Returns:
        JSON response
    """
    try:
        # Get form data
        user_msg = request.form.get('message', '').strip()
        target_agent_req = request.form.get('target_agent', 'GENEL')
        
        # Handle file uploads
        file_path = _handle_file_upload(request.files.get('file'))
        
        # Validate input
        if not user_msg and not file_path:
            return jsonify({
                "status": "error",
                "reply": "Mesaj içeriği boş"
            }), 400
        
        # Handle special commands
        if user_msg.lower() in ["hafızayı sil", "hafızayı temizle"]:
            return _clear_memory()
        
        # Security check
        sec_result = _perform_security_check(request.files.get('auth_frame'))
        
        # Determine if group call
        is_group_call = _is_group_call(user_msg, target_agent_req)
        
        # Process request
        if is_group_call:
            return _handle_team_response(user_msg, sec_result)
        else:
            return _handle_single_response(
                user_msg,
                target_agent_req,
                sec_result,
                file_path
            )
    
    except Exception as e:
        logger.error(f"Web chat hatası: {e}")
        return jsonify({
            "status": "error",
            "reply": f"Sistem hatası: {str(e)[:100]}"
        }), 500


# ═══════════════════════════════════════════════════════════════
# WEBHOOK
# ═══════════════════════════════════════════════════════════════
@app.route('/webhook', methods=['GET', 'POST'])
def webhook_handler() -> Response:
    """
    Meta webhook endpoint
    
    GET: Webhook verification
    POST: Incoming messages
    
    Returns:
        Response
    """
    if request.method == 'GET':
        return _verify_webhook()
    else:
        return _handle_webhook_post()


def _verify_webhook() -> Response:
    """Webhook verification"""
    verify_token = os.getenv(
        "WEBHOOK_VERIFY_TOKEN",
        "lotus_ai_guvenlik_tokeni"
    )
    
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    if mode == "subscribe" and token == verify_token:
        logger.info("✅ Webhook verified")
        return challenge, 200
    
    logger.warning("❌ Webhook verification failed")
    return "Verification failed", 403


def _handle_webhook_post() -> Response:
    """Handle webhook POST"""
    try:
        data = request.json
        
        if not RuntimeContext.messaging_manager:
            return jsonify({"status": "ignored"}), 200
        
        # Parse incoming message
        parsed = RuntimeContext.messaging_manager.parse_incoming_webhook(data)
        
        if parsed:
            RuntimeContext.msg_queue.put(parsed)
            return jsonify({"status": "ok"}), 200
        
        return jsonify({"status": "ignored"}), 200
    
    except Exception as e:
        logger.error(f"Webhook hatası: {e}")
        return jsonify({"status": "error"}), 500


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def _handle_file_upload(file) -> Optional[str]:
    """
    Dosya yükleme
    
    Args:
        file: Uploaded file
    
    Returns:
        Dosya yolu veya None
    """
    if not file or file.filename == '':
        return None
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    
    except Exception as e:
        logger.error(f"Dosya upload hatası: {e}")
        return None


def _clear_memory() -> Response:
    """Hafızayı temizle"""
    if RuntimeContext.engine and RuntimeContext.engine.memory:
        RuntimeContext.engine.memory.clear_history()
        return jsonify({
            "status": "success",
            "agent": "SİSTEM",
            "reply": "Hafıza başarıyla temizlendi"
        })
    
    return jsonify({
        "status": "error",
        "reply": "Hafıza modülü hazır değil"
    }), 503


def _perform_security_check(auth_file) -> tuple:
    """
    Güvenlik kontrolü
    
    Args:
        auth_file: Auth frame file
    
    Returns:
        Security result tuple
    """
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
            logger.error(f"Auth hatası: {e}")
    
    # Determine result
    if identified_user:
        return ("ONAYLI", identified_user, None)
    elif frame_present:
        return ("SORGULAMA", {"name": "Yabancı", "level": 0}, "TANIŞMA_MODU")
    else:
        return ("SORGULAMA", {"name": "Web Kullanıcısı", "level": 1}, "KAMERA_YOK")


def _is_group_call(user_msg: str, target_agent: str) -> bool:
    """
    Group call kontrolü
    
    Args:
        user_msg: Kullanıcı mesajı
        target_agent: Hedef agent
    
    Returns:
        Group call ise True
    """
    group_triggers = [
        "millet", "ekip", "herkes", "gençler",
        "arkadaşlar", "team"
    ]
    
    return (
        target_agent == "GENEL" and
        any(t in user_msg.lower() for t in group_triggers)
    )


def _handle_team_response(user_msg: str, sec_result: tuple) -> Response:
    """
    Team response
    
    Args:
        user_msg: Kullanıcı mesajı
        sec_result: Security result
    
    Returns:
        JSON response
    """
    if not RuntimeContext.engine or not RuntimeContext.loop:
        return jsonify({
            "status": "error",
            "reply": "Motor hazır değil"
        }), 503
    
    try:
        future = asyncio.run_coroutine_threadsafe(
            RuntimeContext.engine.get_team_response(user_msg, sec_result),
            RuntimeContext.loop
        )
        
        replies_list = future.result(timeout=120)
        
        return jsonify({
            "status": "success",
            "replies": replies_list
        })
    
    except asyncio.TimeoutError:
        return jsonify({
            "status": "error",
            "reply": "İşlem zaman aşımına uğradı"
        }), 504
    
    except Exception as e:
        logger.error(f"Team response hatası: {e}")
        return jsonify({
            "status": "error",
            "reply": "İşlem başarısız"
        }), 500


def _handle_single_response(
    user_msg: str,
    target_agent_req: str,
    sec_result: tuple,
    file_path: Optional[str]
) -> Response:
    """
    Single agent response
    
    Args:
        user_msg: Kullanıcı mesajı
        target_agent_req: İstenen agent
        sec_result: Security result
        file_path: Dosya yolu
    
    Returns:
        JSON response
    """
    # Determine target agent
    final_agent = _determine_target_agent(user_msg, target_agent_req)
    
    # Update active agent
    RuntimeContext.active_web_agent = final_agent
    
    # Check if loop is running
    if not RuntimeContext.loop or not RuntimeContext.loop.is_running():
        return jsonify({
            "status": "error",
            "reply": "Motor hazır değil"
        }), 503
    
    try:
        # Get response
        future = asyncio.run_coroutine_threadsafe(
            RuntimeContext.engine.get_response(
                final_agent,
                user_msg,
                sec_result,
                file_path=file_path
            ),
            RuntimeContext.loop
        )
        
        response_data = future.result(timeout=90)
        
        # Play voice if enabled
        if RuntimeContext.voice_mode_active:
            RuntimeContext.executor.submit(
                play_voice,
                response_data['content'],
                response_data['agent'],
                RuntimeContext.state_manager
            )
        
        return jsonify({
            "status": "success",
            "agent": response_data['agent'],
            "reply": response_data['content']
        })
    
    except asyncio.TimeoutError:
        return jsonify({
            "status": "error",
            "reply": "İşlem zaman aşımına uğradı"
        }), 504
    
    except Exception as e:
        logger.error(f"Single response hatası: {e}")
        return jsonify({
            "status": "error",
            "reply": "İşlem başarısız"
        }), 500


def _determine_target_agent(user_msg: str, target_agent_req: str) -> str:
    """
    Hedef agent belirle
    
    Args:
        user_msg: Kullanıcı mesajı
        target_agent_req: İstenen agent
    
    Returns:
        Agent adı
    """
    # Use active web agent as default
    final_agent = RuntimeContext.active_web_agent
    
    # Check if specific agent requested
    if target_agent_req != "GENEL" and target_agent_req in AGENTS_CONFIG:
        return target_agent_req
    
    # Auto-detect agent
    if RuntimeContext.engine:
        detected_agent = RuntimeContext.engine.determine_agent(user_msg)
        if detected_agent:
            return detected_agent
    
    # Fallback to ATLAS
    return "ATLAS"


# ═══════════════════════════════════════════════════════════════
# SERVER RUNNER
# ═══════════════════════════════════════════════════════════════
def run_flask() -> None:
    """
    Flask sunucusu başlat
    
    Runs on 0.0.0.0:5000
    """
    try:
        # Suppress werkzeug logs
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Run server
        app.run(
            host='0.0.0.0',
            port=5000,
            use_reloader=False,
            threaded=True
        )
    
    except Exception as e:
        logger.error(f"Flask sunucu hatası: {e}")