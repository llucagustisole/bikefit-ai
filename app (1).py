import os
import cv2
import json
import tempfile
import numpy as np
import streamlit as st
import mediapipe as mp
from io import BytesIO
from scipy.signal import find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import anthropic

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import urllib.request

# Descargar modelo de MediaPipe automáticamente si no está en el servidor
model_path = 'pose_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN, CONSTANTES Y ESTADO DE SESIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="BikeFit AI", page_icon="🚴", layout="wide")

KP_BIKEFIT = {
    "hombro": 12, "codo": 14, "muneca": 16, "cadera": 24,
    "rodilla": 26, "tobillo": 28, "talon": 30, "pie": 32,
}

CONEXIONES_BIKEFIT = [
    ("hombro", "codo"), ("codo", "muneca"), ("hombro", "cadera"),
    ("cadera", "rodilla"), ("rodilla", "tobillo"), ("tobillo", "talon"), ("tobillo", "pie"),
]

RANGOS = {
    "PMI": {"rodilla": (145.0, 155.0), "cadera": (100.0, 115.0), "tobillo": (95.0, 105.0), "codo": (20.0, 30.0), "torso": (30.0, 55.0)},
    "PMS": {"rodilla": (70.0, 85.0), "cadera": (55.0, 65.0), "tobillo": (75.0, 90.0), "codo": (20.0, 30.0), "torso": (30.0, 55.0)}
}

if "resultados_analisis" not in st.session_state:
    st.session_state.resultados_analisis = None
if "video_procesado" not in st.session_state:
    st.session_state.video_procesado = None

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE CÁLCULO Y DIBUJO
# -----------------------------------------------------------------------------
def calcular_angulo(p1, p2, p3):
    a, b, c = np.array(p1, dtype=float), np.array(p2, dtype=float), np.array(p3, dtype=float)
    ba, bc = a - b, c - b
    norma = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norma < 1e-8: return 0.0
    coseno = np.clip(np.dot(ba, bc) / norma, -1.0, 1.0)
    return round(np.degrees(np.arccos(coseno)), 1)

def draw_landmarks_on_image(rgb_image, detection_result, fase="PMI"):
    if not detection_result or not detection_result.pose_landmarks:
        return np.copy(rgb_image), np.copy(rgb_image), {}

    landmarks = detection_result.pose_landmarks[0]
    annotated_clean = np.copy(rgb_image)
    h, w = annotated_clean.shape[:2]

    def px(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    puntos_px = {nombre: px(idx) for nombre, idx in KP_BIKEFIT.items()}

    for nombre, punto in puntos_px.items():
        vis = landmarks[KP_BIKEFIT[nombre]].visibility
        color_punto = (0, 255, 255) if vis > 0.5 else (0, 80, 255)
        cv2.circle(annotated_clean, punto, 6, color_punto, 2)

    annotated_hud = np.copy(annotated_clean)
    for p1, p2 in CONEXIONES_BIKEFIT:
        cv2.line(annotated_hud, puntos_px[p1], puntos_px[p2], (255, 255, 255), 3)

    hombro, codo, muneca = puntos_px["hombro"], puntos_px["codo"], puntos_px["muneca"]
    cadera, rodilla, tobillo = puntos_px["cadera"], puntos_px["rodilla"], puntos_px["tobillo"]
    pie, talon = puntos_px["pie"], puntos_px["talon"]

    vis_pie = landmarks[KP_BIKEFIT["pie"]].visibility
    vis_talon = landmarks[KP_BIKEFIT["talon"]].visibility
    ref_pie = pie if vis_pie >= vis_talon else talon

    flexion_codo = round(180.0 - calcular_angulo(hombro, codo, muneca), 1)
    dx_torso = abs(hombro[0] - cadera[0])
    dy_torso = abs(cadera[1] - hombro[1])
    angulo_torso_val = round(np.degrees(np.arctan2(dx_torso, dy_torso)), 1)

    angulos = {
        "rodilla": calcular_angulo(cadera, rodilla, tobillo),
        "cadera": calcular_angulo(hombro, cadera, rodilla),
        "tobillo": calcular_angulo(rodilla, tobillo, ref_pie),
        "codo": flexion_codo,
        "torso": angulo_torso_val
    }

    posiciones = {"rodilla": rodilla, "cadera": cadera, "tobillo": tobillo, "codo": codo, "torso": hombro}

    for articulacion, (px_pos, py_pos) in posiciones.items():
        valor = angulos[articulacion]
        mn, mx = RANGOS[fase][articulacion]
        color = (0, 255, 0) if mn <= valor <= mx else (0, 140, 255)
        label_map = {"codo": "Codo (Flex)", "torso": "Torso"}
        texto = f"{label_map.get(articulacion, articulacion.capitalize())}: {valor}deg"

        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated_hud, (px_pos + 15, py_pos - th - 5), (px_pos + 15 + tw, py_pos + 5), (30, 30, 30), -1)
        cv2.putText(annotated_hud, texto, (px_pos + 15, py_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated_clean, annotated_hud, angulos

def generar_informe_bikefit(angulos, rangos, api_key_user=""):
    # Prioridad: 1. Entrada manual en la app | 2. Secrets de Streamlit | 3. Variable de entorno
    api_key = api_key_user.strip() or st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("No se ha encontrado ninguna API Key válida.")
    client = anthropic.Anthropic(api_key=api_key)
    
    def estado_angulo(fase, art):
        val = angulos[fase][art]
        mn, mx = rangos[fase][art]
        if val < mn: return f"{val}° BAJO"
        if val > mx: return f"{val}° ALTO"
        return f"{val}° OPTIMO"

    prompt = f"""Eres un experto en bike fitting biomecánico. Analiza la telemetría de un ciclista en rodillo obtenida mediante visión artificial (MediaPipe).

1. PUNTO MUERTO INFERIOR (PMI) - Máxima Extensión:
- Rodilla: {estado_angulo("PMI", "rodilla")} (Ref: 145-155°)
- Tobillo: {estado_angulo("PMI", "tobillo")} (Ref: 95-105°)
- Codo (Flexión): {estado_angulo("PMI", "codo")} (Ref: 20-30° Pruitt & Matheny, 2006)
- Torso: {estado_angulo("PMI", "torso")} (Ref: Competición 30-45° / Recreativo 45-55° Pruitt & Matheny, 2006)

2. PUNTO MUERTO SUPERIOR (PMS) - Máxima Compresión:
- Cadera: {estado_angulo("PMS", "cadera")} (Ref: 55-65° según Phil Burt, 2014. Nota: Ángulos <55° indican cadera muy cerrada/comprimida; ángulos >65° indican cadera abierta).
- Codo (Flexión): {estado_angulo("PMS", "codo")} (Ref: 20-30° Pruitt & Matheny, 2006)
- Torso: {estado_angulo("PMS", "torso")} (Ref: Competición 30-45° / Recreativo 45-55° Pruitt & Matheny, 2006)

Genera un informe estructurado ÚNICAMENTE en formato JSON con el siguiente esquema exacto:
{{
    "diagnostico_general": "Diagnóstico general",
    "analisis_tren_superior": "Implicaciones del ángulo del torso y codos según Pruitt & Matheny (2006)",
    "recomendaciones_ajuste": "Altura/retroceso del sillín, longitud de potencia o altura del manillar (reach/stack) en mm",
    "prioridad_accion": "Prioridad de acción"
}}

Sé conciso, directo y técnico. Responde exclusivamente con el objeto JSON, sin introducciones ni marcas markdown."""

    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    texto_respuesta = mensaje.content[0].text.strip()
    if texto_respuesta.startswith("```json"): texto_respuesta = texto_respuesta[7:]
    if texto_respuesta.startswith("```"): texto_respuesta = texto_respuesta[3:]
    if texto_respuesta.endswith("```"): texto_respuesta = texto_respuesta[:-3]

    return json.loads(texto_respuesta.strip())

# -----------------------------------------------------------------------------
# 3. GENERADOR DE PDF EN MEMORIA
# -----------------------------------------------------------------------------
def crear_pdf_reporte(img_hud_pmi, img_hud_pms, informe_json):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#1E293B'), spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#0F172A'), spaceBefore=8, spaceAfter=4)
    text_style = ParagraphStyle('TextStyle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#334155'))

    story.append(Paragraph("🚴 Reporte Biomecánico - BikeFit AI", title_style))
    story.append(Spacer(1, 10))

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f_pmi, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f_pms:
        
        cv2.imwrite(f_pmi.name, cv2.cvtColor(img_hud_pmi, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f_pms.name, cv2.cvtColor(img_hud_pms, cv2.COLOR_RGB2BGR))

        img1 = Image(f_pmi.name, width=250, height=180)
        img2 = Image(f_pms.name, width=250, height=180)

        data_imgs = [
            [Paragraph("<b>Punto Muerto Inferior (PMI)</b>", text_style), Paragraph("<b>Punto Muerto Superior (PMS)</b>", text_style)],
            [img1, img2]
        ]
        t_imgs = Table(data_imgs, colWidths=[260, 260])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_imgs)

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Diagnóstico y Recomendaciones (IA)</b>", heading_style))

    data_informe = [
        [Paragraph("<b>Diagnóstico General:</b>", text_style), Paragraph(informe_json.get("diagnostico_general", ""), text_style)],
        [Paragraph("<b>Tren Superior:</b>", text_style), Paragraph(informe_json.get("analisis_tren_superior", ""), text_style)],
        [Paragraph("<b>Ajustes Recomendados:</b>", text_style), Paragraph(informe_json.get("recomendaciones_ajuste", ""), text_style)],
        [Paragraph("<b>Prioridad:</b>", text_style), Paragraph(informe_json.get("prioridad_accion", ""), text_style)]
    ]

    t_info = Table(data_informe, colWidths=[130, 390])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    story.append(t_info)
    doc.build(story)

    os.remove(f_pmi.name)
    os.remove(f_pms.name)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# -----------------------------------------------------------------------------
# 4. INTERFAZ Y FLUJO PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🚴 BikeFit AI - Análisis Biomecánico")

# Guía de grabación directa en la pantalla inicial
with st.expander("📋 Instrucciones de Grabación para un Análisis Preciso", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        * **Plano Perpendicular (Sagital):** Coloca la cámara a 90° exactos del costado del ciclista, a la altura de la caja de pedalier. Evita ángulos picados o inclinados.
        * **Cámara Fija:** Utiliza trípode o un soporte estable. No muevas ni persigas al ciclista durante la grabación.
        """)
    with col_b:
        st.markdown("""
        * **Iluminación y Contraste:** Asegura buena luz sobre el ciclista. Usa ropa ajustada que contraste claramente con el fondo para facilitar el tracking de la visión artificial.
        * **Formato y Duración:** Graba en horizontal. Un vídeo de 5 a 10 segundos pedaleando a cadencia constante de rodillo es suficiente.
        """)

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input("Anthropic API Key", type="password", help="Formato: sk-ant-api03-...")

uploaded_file = st.file_uploader("Sube tu vídeo de análisis en plano sagital", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.video(uploaded_file)
    
    if not api_key_input:
        st.warning("⚠️ Introduce tu API Key de Anthropic en la barra lateral para continuar con el análisis.")
    else:
        if st.session_state.video_procesado != uploaded_file.name:
            st.session_state.resultados_analisis = None

        if st.button("🚀 Iniciar Análisis Biomecánico") or st.session_state.resultados_analisis is not None:
            
            if st.session_state.resultados_analisis is None:
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name

                    model_path = 'pose_landmarker.task'
                    if not os.path.exists(model_path):
                        st.error(f"❌ Falta el archivo del modelo `{model_path}`.")
                        st.stop()

                    status_text = st.empty()
                    progress_bar = st.progress(0)

                    status_text.info("🔎 Procesando fotogramas con MediaPipe...")

                    BaseOptions = mp.tasks.BaseOptions
                    PoseLandmarker = mp.tasks.vision.PoseLandmarker
                    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                    RunningMode = mp.tasks.vision.RunningMode

                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=RunningMode.VIDEO,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.75
                    )
                    detector = PoseLandmarker.create_from_options(options)

                    cap = cv2.VideoCapture(video_path)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0: fps = 30.0

                    datos_video = []
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        timestamp_ms = int(frame_idx * (1000 / fps))
                        resultado = detector.detect_for_video(mp_image, timestamp_ms)

                        if resultado.pose_landmarks:
                            landmarks = resultado.pose_landmarks[0]
                            datos_video.append({
                                "frame": frame_idx,
                                "tobillo_y": int(landmarks[28].y * h),
                                "resultado": resultado
                            })
                        frame_idx += 1

                    cap.release()
                    detector.close()
                    progress_bar.progress(50)

                    if not datos_video:
                        st.error("❌ No se identificaron puntos anatómicos en el vídeo. Revisa el ángulo de la cámara o la iluminación.")
                        st.stop()

                    status_text.info("📐 Detectando PMI / PMS y midiendo ángulos...")

                    serie_y_tobillo = [d["tobillo_y"] for d in datos_video]

                    indices_pmi, _ = find_peaks(serie_y_tobillo, prominence=20.0, distance=15)
                    pmi_ganador = datos_video[np.argmax(serie_y_tobillo)] if len(indices_pmi) == 0 else datos_video[min(indices_pmi, key=lambda idx: abs(serie_y_tobillo[idx] - np.median([serie_y_tobillo[i] for i in indices_pmi])))]

                    serie_y_invertida = [-y for y in serie_y_tobillo]
                    indices_pms, _ = find_peaks(serie_y_invertida, prominence=20.0, distance=15)
                    pms_ganador = datos_video[np.argmin(serie_y_tobillo)] if len(indices_pms) == 0 else datos_video[min(indices_pms, key=lambda idx: abs(serie_y_tobillo[idx] - np.median([serie_y_tobillo[i] for i in indices_pms])))]

                    cap = cv2.VideoCapture(video_path)
                    frame_pmi, frame_pms = None, None
                    idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        if idx == pmi_ganador["frame"]: frame_pmi = frame.copy()
                        if idx == pms_ganador["frame"]: frame_pms = frame.copy()
                        if frame_pmi is not None and frame_pms is not None: break
                        idx += 1
                    cap.release()

                    rgb_pmi = cv2.cvtColor(frame_pmi, cv2.COLOR_BGR2RGB)
                    clean_pmi, hud_pmi, angulos_pmi = draw_landmarks_on_image(rgb_pmi, pmi_ganador["resultado"], fase="PMI")

                    rgb_pms = cv2.cvtColor(frame_pms, cv2.COLOR_BGR2RGB)
                    clean_pms, hud_pms, angulos_pms = draw_landmarks_on_image(rgb_pms, pms_ganador["resultado"], fase="PMS")

                    angulos_calculados = {"PMI": angulos_pmi, "PMS": angulos_pms}

                    status_text.info("🤖 Consultando informe técnico a Claude API...")
                    informe = generar_informe_bikefit(angulos_calculados, RANGOS, api_key_input)
                    pdf_data = crear_pdf_reporte(hud_pmi, hud_pms, informe)

                    progress_bar.progress(100)
                    status_text.empty()

                    st.session_state.resultados_analisis = {
                        "clean_pmi": clean_pmi, "hud_pmi": hud_pmi, "frame_pmi": pmi_ganador["frame"],
                        "clean_pms": clean_pms, "hud_pms": hud_pms, "frame_pms": pms_ganador["frame"],
                        "informe": informe, "pdf_data": pdf_data
                    }
                    st.session_state.video_procesado = uploaded_file.name

                except Exception as e:
                    st.error(f"❌ Error durante el procesado: {e}")
                    st.stop()

            res = st.session_state.resultados_analisis
            tab1, tab2, tab3 = st.tabs(["📊 PMI (Extensión)", "📊 PMS (Compresión)", "🤖 Informe IA"])

            with tab1:
                st.subheader(f"Punto Muerto Inferior (Frame {res['frame_pmi']})")
                c1, c2 = st.columns(2)
                c1.image(res["clean_pmi"], caption="Captura Limpia", use_container_width=True)
                c2.image(res["hud_pmi"], caption="Telemetría", use_container_width=True)

            with tab2:
                st.subheader(f"Punto Muerto Superior (Frame {res['frame_pms']})")
                c1, c2 = st.columns(2)
                c1.image(res["clean_pms"], caption="Captura Limpia", use_container_width=True)
                c2.image(res["hud_pms"], caption="Telemetría", use_container_width=True)

            with tab3:
                st.subheader("Informe Biomecánico Generado")
                st.json(res["informe"])

                st.download_button(
                    label="📄 Descargar Informe Completo en PDF",
                    data=res["pdf_data"],
                    file_name="informe_bikefit_ai.pdf",
                    mime="application/pdf"
                )
