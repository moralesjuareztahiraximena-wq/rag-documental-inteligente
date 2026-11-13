# RAG Documental Público — prototipo

Aplicación pública de consulta de PDF, DOCX y TXT. Los visitantes pueden consultar los fragmentos indexados; solo el administrador puede cargar o eliminar archivos. El acceso público utiliza búsqueda TF-IDF, **no respuestas generativas**. La generación con IA está reservada al administrador, si configura una API.

## Publicar en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube `app.py`, `requirements.txt` y `.gitignore` (no subas documentos privados).
2. En https://share.streamlit.io elige Create app y selecciona el repositorio y `app.py`.
3. En Advanced settings → Secrets configura `ADMIN_PASSWORD = "una-clave-larga-unica"` y, opcionalmente, `OPENAI_API_KEY = "..."`. No incluyas secretos en GitHub.
4. Pulsa Deploy y prueba la dirección generada en modo incógnito.

## Seguridad y limitaciones

- **No publiques documentos privados, personales o con derechos de difusión restringidos.** Los fragmentos indexados serán visibles a visitantes.
- La biblioteca usa SQLite en disco local. En Community Cloud puede perderse al reiniciar/recrear el servidor; no es almacenamiento persistente garantizado. Para producción, migrar a una base gestionada y almacenamiento de objetos.
- Los visitantes no pueden generar respuestas IA, para evitar consumo ilimitado de API. Para ofrecer IA pública, implementar cuotas por usuario/IP, autenticación opcional, límites de gasto y monitoreo.
- Los PDF escaneados requieren OCR, no incluido.
- Las normas cargadas pueden estar desactualizadas; verificar vigencia en fuentes oficiales.
- Antes de producción real, implementar límites de solicitudes en servidor, antivirus/validación de archivos, política de privacidad, registro de actividad y copias de seguridad.
