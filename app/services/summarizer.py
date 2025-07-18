# app/services/summarizer.py
from transformers import pipeline
import torch
import re
from typing import Dict, List, Optional

MAX_INPUT_CHARS = 3000  # máximo caracteres para o modelo
device = 0 if torch.cuda.is_available() else -1

# Inicializa pipeline de sumarização (modelo genérico, balanceado)
summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device,
    max_length=512,
    truncation=True
)


def clean_text(text: str) -> str:
    """Limpa texto removendo caracteres estranhos e normalizando espaços."""
    text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\@\+\#\&\%\àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]', '', text)  # noqa: E501
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('\n')
    filtered = [line.strip() for line in lines if len(line.strip()) > 3]
    return ' '.join(filtered).strip()


def extract_structured_info(text: str) -> Dict[str, List[str]]:
    """
    Extrai seções principais do currículo, incluindo:
    - experiência profissional
    - formação acadêmica
    - habilidades (técnicas e comportamentais)
    - certificações/cursos
    - conquistas/resultados (opcional)
    """
    section_patterns = {
        'experiencia': [
            r'experiência\s*profissional.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'experiência.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'trabalho.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'carreira.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'histórico\s*profissional.*?(?=\n\s*\n|\n[A-Z]|$)'
        ],
        'formacao': [
            r'formação.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'educação.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'acadêmica.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'ensino.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'graduação.*?(?=\n\s*\n|\n[A-Z]|$)'
        ],
        'habilidades': [
            r'habilidades.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'competências.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'skills.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'habilidades comportamentais.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'soft skills.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'tecnologias.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'conhecimentos.*?(?=\n\s*\n|\n[A-Z]|$)'
        ],
        'certificacoes': [
            r'certificações.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'certificados.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'cursos.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'qualificações.*?(?=\n\s*\n|\n[A-Z]|$)'
        ],
        'conquistas': [
            r'conquistas.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'resultados.*?(?=\n\s*\n|\n[A-Z]|$)',
            r'prêmios.*?(?=\n\s*\n|\n[A-Z]|$)'
        ]
    }

    extracted_info = {}
    text_lower = text.lower()

    for section, patterns in section_patterns.items():
        contents = []
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL)  # noqa: E501
            for m in matches:
                c = m.group(0).strip()
                if len(c) > 20:
                    contents.append(c)
        if contents:
            extracted_info[section] = contents

    return extracted_info


def extract_experience_years(text: str) -> Optional[str]:
    """Extrai anos de experiência geral do texto."""
    patterns = [
        r'(\d+)\s*anos?\s*de\s*experiência',
        r'experiência\s*de\s*(\d+)\s*anos?',
        r'(\d+)\+?\s*anos?\s*experiência',
        r'mais\s*de\s*(\d+)\s*anos?',
        r'(\d+)\s*years?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\s*years?'
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return f"{m.group(1)} anos de experiência"
    return None


def extract_job_titles(text: str) -> List[str]:
    """Extrai cargos/funções amplas (genéricas, para várias áreas)."""
    title_pattern = r'(gerente|coordenador|analista|assistente|técnico|supervisor|consultor|especialista|diretor|presidente|administrador|engenheiro|desenvolvedor|programador|professor|médico|enfermeiro|advogado|contador|vendedor|representante|estagiário|estagiaria|auxiliar|operador)(?:\s+de\s+[\w\s]+)?'  # noqa: E501
    titles = re.findall(title_pattern, text.lower())
    return list(set(titles))  # Remove duplicatas


def create_structured_summary(text: str) -> str:
    """Cria resumo estruturado focado em aptidão para vaga, com base em extrações."""  # noqa: E501
    info = extract_structured_info(text)
    experience = extract_experience_years(text)
    titles = extract_job_titles(text)

    summary_parts = []

    if titles:
        cargos_str = ', '.join([t.title() for t in titles])
        summary_parts.append(f"Cargos/Funções: {cargos_str}")

    if experience:
        summary_parts.append(f"Experiência: {experience}")

    if 'habilidades' in info:
        habilidades = ' '.join(info['habilidades'])
        habilidades = habilidades[:200] + ('...' if len(habilidades) > 200 else '')  # noqa: E501
        summary_parts.append(f"Competências e habilidades: {habilidades}")

    if 'certificacoes' in info:
        certs = ' '.join(info['certificacoes'])
        certs = certs[:200] + ('...' if len(certs) > 200 else '')
        summary_parts.append(f"Certificações e cursos: {certs}")

    if 'conquistas' in info:
        conquistas = ' '.join(info['conquistas'])
        conquistas = conquistas[:200] + ('...' if len(conquistas) > 200 else '')  # noqa: E501
        summary_parts.append(f"Conquistas/Resultados: {conquistas}")

    if 'formacao' in info or any(k in text.lower() for k in ['superior', 'graduação', 'bacharelado', 'tecnólogo', 'técnico']):  # noqa: E501
        formacao_str = 'Formação acadêmica mencionada'
        summary_parts.append(formacao_str)

    # Monta resumo
    if summary_parts:
        result = ". ".join(summary_parts) + "."
        # Normaliza espaços e remove repetições bobas
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    # Fallback genérico
    return "Profissional com experiência e formação diversificada."


def summarize_text(text: str) -> str:
    """Função principal para gerar resumo do currículo."""
    if not text.strip():
        return "Texto vazio, não foi possível gerar resumo."

    cleaned = clean_text(text)

    if len(cleaned) < 100:
        return "Texto muito curto para gerar resumo adequado."

    try:
        structured_summary = create_structured_summary(cleaned)

        # Se resumo estruturado é satisfatório, retorna
        if len(structured_summary) > 80:
            return structured_summary

        # Caso contrário, usa modelo ML para gerar resumo mais natural
        input_for_model = cleaned[:MAX_INPUT_CHARS]

        ml_summary = summarizer_pipeline(
            input_for_model,
            max_length=180,
            min_length=50,
            do_sample=False,
            num_beams=3,
            early_stopping=True
        )[0]["summary_text"]

        # Se resumo estruturado for muito curto, retorna só ML
        if len(structured_summary) < 50:
            final = ml_summary
        else:
            # Tenta combinar ambos, dando prioridade ao estruturado
            final = f"{structured_summary} {ml_summary}"

        final = re.sub(r'\s+', ' ', final).strip()
        if not final.endswith('.'):
            final += '.'
        return final

    except Exception as e:
        try:
            return create_structured_summary(cleaned)
        except:   # noqa: E722
            return f"Erro ao gerar resumo: {str(e)}"


def analyze_resume_details(text: str) -> Dict[str, any]:
    """Retorna análise detalhada opcional do currículo."""
    cleaned = clean_text(text)
    return {
        'summary': summarize_text(text),
        'experience_years': extract_experience_years(cleaned),
        'job_titles': extract_job_titles(cleaned),
        'sections_found': list(extract_structured_info(cleaned).keys()),
        'text_length': len(cleaned),
        'word_count': len(cleaned.split())
    }
