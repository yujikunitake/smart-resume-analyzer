from PIL import Image
import easyocr
import io
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa o OCR em português e inglês
reader = easyocr.Reader(['pt', 'en'])


def preprocess_image(image_array):
    """Converte imagem para escala de cinza e redimensiona se necessário"""
    image = Image.fromarray(image_array) if isinstance(image_array, np.ndarray) else image_array  # noqa: E501

    if image.mode != 'L':
        image = image.convert('L')

    width, height = image.size
    if width > 2000 or height > 2000:
        ratio = min(2000 / width, 2000 / height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)  # noqa: E501

    return np.array(image)


def extract_text_from_image(image_array):
    """Executa OCR com EasyOCR em uma imagem"""
    try:
        processed_image = preprocess_image(image_array)

        result = reader.readtext(
            processed_image,
            paragraph=True,
            detail=0,
            width_ths=0.9,
            height_ths=0.9
        )

        text = " ".join(result) if isinstance(result, list) else str(result)
        return text.strip()

    except Exception as e:
        logger.error(f"Erro no OCR da imagem: {str(e)}")
        return f"[ERRO OCR]: {str(e)}"


def extract_text(filename: str, content_bytes: bytes) -> str:
    """
    Extrai texto de arquivos enviados (PDFs ou imagens).
    Não aplica filtro de qualidade — retorna todo o texto possível.
    """
    try:
        logger.info(f"Iniciando extração de: {filename}")

        if filename.lower().endswith(".pdf"):
            try:
                images = convert_from_bytes(
                    content_bytes,
                    dpi=300,
                    first_page=1,
                    last_page=10
                )

                if not images:
                    return "[ERRO]: PDF sem páginas processáveis."

                full_text = ""
                for idx, image in enumerate(images):
                    logger.info(f"Página {idx+1} de {len(images)}")
                    image_array = np.array(image)
                    page_text = extract_text_from_image(image_array)

                    if page_text:
                        full_text += page_text + "\n\n"
                    else:
                        logger.warning(f"Nenhum texto extraído na página {idx+1}")  # noqa: E501

                return full_text.strip() or "[AVISO]: Nenhum texto extraído do PDF."  # noqa: E501

            except PDFPageCountError:
                return "[ERRO]: PDF vazio ou inválido."
            except Exception as e:
                logger.error(f"Erro no processamento do PDF: {str(e)}")
                return f"[ERRO PDF]: {str(e)}"

        else:
            try:
                image = Image.open(io.BytesIO(content_bytes))

                if image.mode == 'RGBA':
                    image = image.convert('RGB')

                image_array = np.array(image)
                text = extract_text_from_image(image_array)

                return text.strip() or "[AVISO]: Nenhum texto extraído da imagem."  # noqa: E501

            except Exception as e:
                logger.error(f"Erro no processamento da imagem: {str(e)}")
                return f"[ERRO IMAGEM]: {str(e)}"

    except Exception as e:
        logger.error(f"Erro geral na extração: {str(e)}")
        return f"[ERRO GERAL]: {str(e)}"
