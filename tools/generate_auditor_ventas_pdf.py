from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Guia_Inicio_Auditor_Ventas_VAE.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#1f3a5f"),
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#3f4f63"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1f3a5f"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubsectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#243447"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.5,
            textColor=colors.HexColor("#222222"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#333333"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=11,
            textColor=colors.white,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=10.4,
            textColor=colors.HexColor("#222222"),
        )
    )
    return styles


def p(text, style):
    return Paragraph(text, style)


def bullet(text, styles):
    return p(f"- {text}", styles["Body"])


def draw_header_footer(canvas, doc):
    canvas.saveState()
    width, height = LETTER
    canvas.setStrokeColor(colors.HexColor("#d8dee8"))
    canvas.line(0.65 * inch, height - 0.55 * inch, width - 0.65 * inch, height - 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#58677a"))
    canvas.drawString(0.65 * inch, height - 0.42 * inch, "Auditor de Ventas Inteligente - Guia de arranque")
    canvas.drawRightString(width - 0.65 * inch, 0.42 * inch, f"Pagina {doc.page}")
    canvas.restoreState()


def make_table(rows, col_widths):
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9d2df")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.65 * inch,
        title="Guia de inicio - Auditor de Ventas VAE",
        author="Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=draw_header_footer)])

    story = []

    story.append(Spacer(1, 1.2 * inch))
    story.append(p("Guia de inicio del proyecto", styles["CoverTitle"]))
    story.append(p("Auditor de Ventas Inteligente con VAE", styles["CoverTitle"]))
    story.append(
        p(
            "Plan practico para construir en una semana una solucion basada en redes neuronales "
            "para detectar anomalías en ventas de un restaurante MiPYME ecuatoriano.",
            styles["CoverSubtitle"],
        )
    )
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        p(
            "Enfoque definido: restaurante pequeño con punto de venta, cajeros, meseros, mesas, pedidos, descuentos, anulaciones y devoluciones.",
            styles["CoverSubtitle"],
        )
    )
    story.append(PageBreak())

    story.append(p("1. Idea central", styles["SectionTitle"]))
    story.append(
        p(
            "El proyecto consiste en construir un sistema que aprende el comportamiento normal de las "
            "transacciones de un restaurante ecuatoriano y marca como sospechosas las ventas, devoluciones, "
            "anulaciones, descuentos o cierres de cuenta que se desvían del patrón esperado.",
            styles["Body"],
        )
    )
    story.extend(
        [
            bullet("Problema: perdidas silenciosas por errores de caja, descuentos fuera de politica, anulaciones sospechosas, devoluciones y digitacion incorrecta.", styles),
            bullet("Solucion: un auditor automatico que asigna un score de anomalía a cada transaccion.", styles),
            bullet("RNA usada: Autoencoder Variacional, entrenado principalmente con transacciones normales.", styles),
            bullet("Resultado de negocio: priorizar las transacciones que el dueño o administrador del restaurante debe revisar.", styles),
        ]
    )

    story.append(p("2. Alcance MVP para una semana", styles["SectionTitle"]))
    story.append(
        p(
            "El alcance debe mantenerse pequeño. La meta no es crear un POS completo, sino una herramienta "
            "que demuestre deteccion de anomalías, visualizacion clara y metricas tecnicas.",
            styles["Body"],
        )
    )
    story.extend(
        [
            bullet("Carga de archivo CSV con ventas.", styles),
            bullet("Formulario manual para evaluar una transaccion en vivo.", styles),
            bullet("Modelo VAE entrenado con datos normales.", styles),
            bullet("Tabla de resultados con semaforo: normal, baja, media y alta severidad.", styles),
            bullet("Dashboard con numero de anomalías, monto en riesgo, latencia, throughput y logs.", styles),
            bullet("Prueba con minimo 3 usuarios concurrentes.", styles),
        ]
    )

    story.append(p("3. Datos iniciales", styles["SectionTitle"]))
    story.append(p("Campos recomendados del CSV:", styles["Body"]))
    dataset_rows = [
        [p("Campo", styles["TableHeader"]), p("Descripcion", styles["TableHeader"])],
        [p("id_transaccion", styles["TableCell"]), p("Identificador unico de la venta.", styles["TableCell"])],
        [p("fecha_hora", styles["TableCell"]), p("Fecha y hora de la transaccion.", styles["TableCell"])],
        [p("dia_semana / hora", styles["TableCell"]), p("Variables temporales para detectar patrones por horario.", styles["TableCell"])],
        [p("cajero / mesero", styles["TableCell"]), p("Empleado que registra o atiende la transaccion.", styles["TableCell"])],
        [p("mesa / canal", styles["TableCell"]), p("Mesa, domicilio, retiro en local o app de delivery.", styles["TableCell"])],
        [p("categoria_producto", styles["TableCell"]), p("Categoria: desayunos, almuerzos, bebidas, postres, combos, extras.", styles["TableCell"])],
        [p("monto", styles["TableCell"]), p("Valor de la transaccion.", styles["TableCell"])],
        [p("descuento_pct", styles["TableCell"]), p("Porcentaje de descuento aplicado.", styles["TableCell"])],
        [p("metodo_pago", styles["TableCell"]), p("Efectivo, tarjeta, transferencia u otro.", styles["TableCell"])],
        [p("tipo_transaccion", styles["TableCell"]), p("Venta, devolucion o anulacion.", styles["TableCell"])],
        [p("es_anomalia / tipo_anomalia", styles["TableCell"]), p("Etiqueta para evaluar el modelo en datos sinteticos.", styles["TableCell"])],
    ]
    story.append(make_table(dataset_rows, [1.65 * inch, 4.75 * inch]))

    story.append(p("Anomalias que deben inyectarse", styles["SubsectionTitle"]))
    story.extend(
        [
            bullet("Descuento mayor al 50% sin justificacion.", styles),
            bullet("Venta fuera del horario normal del restaurante.", styles),
            bullet("Anulacion o devolucion de monto alto o muy frecuente.", styles),
            bullet("Monto diez veces mayor al ticket promedio del restaurante.", styles),
            bullet("Monto negativo, cero o inconsistente.", styles),
            bullet("Venta con descuento alto y pago en efectivo.", styles),
            bullet("Cajero o mesero con tasa de anulaciones inusualmente alta.", styles),
        ]
    )

    story.append(p("4. Arquitectura de la solucion", styles["SectionTitle"]))
    arch_rows = [
        [p("Componente", styles["TableHeader"]), p("Responsabilidad", styles["TableHeader"])],
        [p("Preprocesamiento", styles["TableCell"]), p("Normalizar montos, codificar categorias, convertir hora a seno/coseno y preparar el vector de entrada.", styles["TableCell"])],
        [p("VAE", styles["TableCell"]), p("Encoder denso, espacio latente con mu y sigma, reparametrizacion y decoder para reconstruir la transaccion.", styles["TableCell"])],
        [p("Score de anomalia", styles["TableCell"]), p("Error de reconstruccion total, ponderado si se desea dar mas peso a monto y descuento.", styles["TableCell"])],
        [p("Umbral", styles["TableCell"]), p("Percentil 95 del error en validacion normal, ajustable por negocio.", styles["TableCell"])],
        [p("Backend", styles["TableCell"]), p("FastAPI con endpoints para transaccion individual, batch CSV, metricas y logs.", styles["TableCell"])],
        [p("Frontend", styles["TableCell"]), p("Streamlit para carga de CSV, formulario manual, tabla, dashboard y descarga de resultados.", styles["TableCell"])],
    ]
    story.append(make_table(arch_rows, [1.65 * inch, 4.75 * inch]))

    story.append(p("5. Tecnologias a utilizar", styles["SectionTitle"]))
    story.append(
        p(
            "Todas las herramientas propuestas son gratuitas y suficientes para el MVP. La prioridad es "
            "terminar un flujo completo: generar datos, entrenar, evaluar, exponer API, visualizar y medir.",
            styles["Body"],
        )
    )
    tech_rows = [
        [p("Parte", styles["TableHeader"]), p("Tecnologia", styles["TableHeader"]), p("Uso concreto", styles["TableHeader"])],
        [p("Lenguaje base", styles["TableCell"]), p("Python", styles["TableCell"]), p("Implementar generacion de datos, preprocesamiento, entrenamiento, backend y app.", styles["TableCell"])],
        [p("Datos sinteticos", styles["TableCell"]), p("pandas, NumPy", styles["TableCell"]), p("Crear ventas normales y anomalías de restaurante en CSV.", styles["TableCell"])],
        [p("Preprocesamiento", styles["TableCell"]), p("scikit-learn, joblib", styles["TableCell"]), p("One-hot encoding, normalizacion, division train/test y guardado del preprocesador.", styles["TableCell"])],
        [p("RNA / VAE", styles["TableCell"]), p("PyTorch", styles["TableCell"]), p("Construir encoder, reparametrizacion, decoder, funcion de perdida y entrenamiento.", styles["TableCell"])],
        [p("Metricas ML", styles["TableCell"]), p("scikit-learn", styles["TableCell"]), p("Precision, recall, F1, matriz de confusion, ROC-AUC si aplica.", styles["TableCell"])],
        [p("Backend", styles["TableCell"]), p("FastAPI, Uvicorn, Pydantic", styles["TableCell"]), p("Endpoints /predict, /batch y /metrics; validacion de transacciones.", styles["TableCell"])],
        [p("Frontend", styles["TableCell"]), p("Streamlit, Plotly", styles["TableCell"]), p("Carga CSV, formulario manual, tabla con semaforo y dashboard interactivo.", styles["TableCell"])],
        [p("Logs y recursos", styles["TableCell"]), p("logging, time, psutil", styles["TableCell"]), p("Registrar predicciones, latencia, throughput, CPU y RAM.", styles["TableCell"])],
        [p("Concurrencia", styles["TableCell"]), p("Locust o 3 sesiones Streamlit", styles["TableCell"]), p("Demostrar tres usuarios usando la app en paralelo.", styles["TableCell"])],
        [p("Despliegue", styles["TableCell"]), p("Hugging Face Spaces o Render", styles["TableCell"]), p("Publicar el prototipo si hay tiempo; si no, demo local estable.", styles["TableCell"])],
        [p("Presentacion", styles["TableCell"]), p("PowerPoint, Google Slides o Canva", styles["TableCell"]), p("PPT con problema, arquitectura, metricas, demo, negocio, riesgos y pricing.", styles["TableCell"])],
        [p("Control de versiones", styles["TableCell"]), p("Git", styles["TableCell"]), p("Coordinar trabajo del grupo y mantener cambios ordenados.", styles["TableCell"])],
    ]
    story.append(make_table(tech_rows, [1.2 * inch, 1.65 * inch, 3.55 * inch]))

    story.append(p("6. Division del trabajo para 6 personas", styles["SectionTitle"]))
    team_rows = [
        [p("Persona", styles["TableHeader"]), p("Responsabilidad principal", styles["TableHeader"]), p("Entrega minima", styles["TableHeader"])],
        [p("Martín Dávalos", styles["TableCell"]), p("Datos y contexto del restaurante", styles["TableCell"]), p("CSV sintetico, reglas del negocio y tipos de anomalía.", styles["TableCell"])],
        [p("2", styles["TableCell"]), p("Preprocesamiento", styles["TableCell"]), p("Pipeline que transforma CSV a tensores listos para el VAE.", styles["TableCell"])],
        [p("3", styles["TableCell"]), p("Modelo VAE", styles["TableCell"]), p("Entrenamiento, guardado del modelo y error de reconstruccion.", styles["TableCell"])],
        [p("4", styles["TableCell"]), p("Evaluacion y umbrales", styles["TableCell"]), p("Precision, recall, F1, umbral, severidad y monto en riesgo.", styles["TableCell"])],
        [p("5", styles["TableCell"]), p("Backend y metricas tecnicas", styles["TableCell"]), p("Endpoints, logs, latencia, throughput y pruebas concurrentes.", styles["TableCell"])],
        [p("6", styles["TableCell"]), p("Frontend, demo y PPT", styles["TableCell"]), p("Streamlit funcional, guion de demo y diapositivas.", styles["TableCell"])],
    ]
    story.append(make_table(team_rows, [1.25 * inch, 1.85 * inch, 3.3 * inch]))

    story.append(p("7. Plan de 5 dias", styles["SectionTitle"]))
    plan_rows = [
        [p("Dia", styles["TableHeader"]), p("Objetivo", styles["TableHeader"]), p("Resultado esperado", styles["TableHeader"])],
        [p("1", styles["TableCell"]), p("Definir negocio y generar datos.", styles["TableCell"]), p("Dataset normal + anomalías documentadas.", styles["TableCell"])],
        [p("2", styles["TableCell"]), p("Preprocesar y entrenar primer VAE.", styles["TableCell"]), p("Modelo que separa parcialmente normal vs anomalo.", styles["TableCell"])],
        [p("3", styles["TableCell"]), p("Calcular score, umbral y backend.", styles["TableCell"]), p("API evaluando transacciones individuales y por lote.", styles["TableCell"])],
        [p("4", styles["TableCell"]), p("Construir frontend.", styles["TableCell"]), p("Carga CSV, formulario, tabla con semaforo y dashboard.", styles["TableCell"])],
        [p("5", styles["TableCell"]), p("Probar, medir y preparar presentacion.", styles["TableCell"]), p("Demo estable, metricas, logs y PPT terminada.", styles["TableCell"])],
    ]
    story.append(make_table(plan_rows, [0.55 * inch, 2.5 * inch, 3.35 * inch]))

    story.append(PageBreak())

    story.append(p("8. Estructura sugerida del repositorio", styles["SectionTitle"]))
    repo_text = """
<font name="Courier">
RNA_IA/<br/>
  data/<br/>
    ventas_restaurante_sinteticas.csv<br/>
  models/<br/>
    vae_model.pt<br/>
    preprocessor.pkl<br/>
  src/<br/>
    generate_data.py<br/>
    preprocess.py<br/>
    train_vae.py<br/>
    evaluate.py<br/>
    api.py<br/>
    app_streamlit.py<br/>
  requirements.txt<br/>
  reports/<br/>
    metricas_demo.csv<br/>
    logs_predicciones.csv<br/>
  output/pdf/<br/>
    Guia_Inicio_Auditor_Ventas_VAE.pdf
</font>
"""
    story.append(p(repo_text, styles["Small"]))

    story.append(p("9. Metricas para presentar", styles["SectionTitle"]))
    story.extend(
        [
            bullet("Calidad del modelo: precision, recall, F1 y matriz de confusion sobre anomalías conocidas.", styles),
            bullet("Score de anomalía: promedio normal vs promedio anomalo.", styles),
            bullet("Latencia: tiempo promedio para evaluar una transaccion individual.", styles),
            bullet("Throughput: transacciones por segundo en procesamiento batch.", styles),
            bullet("Recursos: CPU y RAM durante inferencia.", styles),
            bullet("Estabilidad: 3 usuarios simultaneos usando la app sin errores.", styles),
            bullet("Negocio: monto total en riesgo detectado y numero de casos priorizados.", styles),
        ]
    )

    story.append(p("10. Modelo de negocio recomendado", styles["SectionTitle"]))
    story.append(
        p(
            "El modelo mas facil de defender para MiPYMEs es una suscripcion mensual por volumen de "
            "transacciones. Es coherente porque el problema ocurre todos los dias, el sistema requiere "
            "mantenimiento, recalibracion de umbrales y soporte, y el cliente evita una inversion inicial alta.",
            styles["Body"],
        )
    )
    pricing_rows = [
        [p("Plan", styles["TableHeader"]), p("Uso sugerido", styles["TableHeader"]), p("Precio referencial", styles["TableHeader"])],
        [p("Micro", styles["TableCell"]), p("Hasta 2.000 transacciones/mes.", styles["TableCell"]), p("USD 9,99 - 14,99/mes.", styles["TableCell"])],
        [p("PYME", styles["TableCell"]), p("Hasta 10.000 transacciones/mes.", styles["TableCell"]), p("USD 24,99 - 39,99/mes.", styles["TableCell"])],
        [p("Pro", styles["TableCell"]), p("Mayor volumen, soporte y reportes.", styles["TableCell"]), p("USD 49,99+/mes.", styles["TableCell"])],
    ]
    story.append(make_table(pricing_rows, [1.25 * inch, 3.05 * inch, 2.1 * inch]))

    story.append(p("11. Riesgos y limitaciones", styles["SectionTitle"]))
    story.extend(
        [
            bullet("Los datos sinteticos no capturan toda la variabilidad de un negocio real.", styles),
            bullet("El sistema sugiere revision humana; no debe bloquear ventas automaticamente.", styles),
            bullet("Puede haber falsos positivos en ventas legitimas pero poco frecuentes.", styles),
            bullet("Cada negocio necesita recalibrar el umbral de anomalía.", styles),
            bullet("Trabajo futuro: entrenar con datos reales anonimizados y agregar explicaciones por campo.", styles),
        ]
    )

    story.append(p("12. Primeros pasos inmediatos", styles["SectionTitle"]))
    story.extend(
        [
            bullet("Fijar el tipo de restaurante: almuerzos, cafeteria, comida rapida o restaurante familiar.", styles),
            bullet("Definir horarios, cajeros, meseros, mesas/canales, categorias y politicas de descuento.", styles),
            bullet("Generar 5.000 a 20.000 transacciones sinteticas.", styles),
            bullet("Separar normales para entrenamiento y anomalías para validacion.", styles),
            bullet("Construir primero el pipeline completo, aunque el modelo inicial sea pequeño.", styles),
        ]
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
