# app/services/question_answering.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
if device >= 0:
    model = model.to(device)

PROMPT_TEMPLATE = """
Analise o currículo abaixo e responda se o candidato se enquadra para a 
pergunta feita.

Currículo:
{curriculo}

Pergunta: {query}

Responda APENAS com "Sim" ou "Não" seguido de uma justificativa específica 
baseada no currículo.

Formato obrigatório:
Sim. [cite tecnologias, experiências e competências específicas do currículo]
ou
Não. [explique o que falta no currículo para atender a pergunta]

Resposta:
"""


def completion(prompt: str, max_length=800):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)  # noqa: E501
    if device >= 0:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        num_beams=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        temperature=0.1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def extract_answer_and_justification(text: str):
    """
    Extrai a resposta (Sim ou Não) e uma justificativa da saída do modelo.
    """
    text = text.strip()

    # Debug: imprimir o texto original para análise
    print(f"Texto original do modelo: '{text}'")

    # Verificar se o modelo retornou o texto dos exemplos/placeholders
    if "[justificativa" in text.lower() or "[cite tecnologias" in text.lower() or "[explique o que falta" in text.lower():  # noqa: E501
        print("Modelo retornou placeholder - tentando fallback")
        return "Erro", "Modelo retornou resposta genérica. Tente novamente."

    # Padrões para capturar resposta e justificativa
    patterns = [
        r'^(Sim|Não|Nao)\.?\s+(.+)',       # Sim/Não seguido de ponto e texto
        r'^(Sim|Não|Nao)[\.:–-]\s*(.+)',   # Sim/Não seguido de pontuação
        r'^(Sim|Não|Nao)\s+(.+)',          # Sim/Não seguido de espaço
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            raw_answer = match.group(1).strip().lower()
            justification = match.group(2).strip()

            # Limpar justificativa
            if justification:
                # Remove pontuação no início se houver
                justification = re.sub(r'^[.:–-]+\s*', '', justification)
                # Remove quebras de linha excessivas e normaliza espaços
                justification = re.sub(r'\s+', ' ', justification)
                justification = justification.strip()

            answer = "Sim" if raw_answer in ["sim", "yes"] else "Não"

            # Validar se a justificativa é válida
            if not justification or len(justification) < 15:
                justification = "Análise baseada no conteúdo do currículo fornecido."  # noqa: E501

            return answer, justification

    # Se não encontrou padrão, tenta uma abordagem mais simples
    if text.lower().startswith('sim'):
        answer = "Sim"
        justification = text[3:].strip()
        if justification.startswith('.'):
            justification = justification[1:].strip()
    elif text.lower().startswith('não') or text.lower().startswith('nao'):
        answer = "Não"
        justification = text[3:].strip()
        if justification.startswith('.'):
            justification = justification[1:].strip()
    else:
        answer = "Não identificado"
        justification = text

    if not justification or len(justification) < 15:
        justification = "Análise baseada no conteúdo do currículo fornecido."

    return answer, justification


def fallback_analysis(context: str, question: str):
    """Análise de fallback baseada em palavras-chave do currículo"""
    context_lower = context.lower()
    question_lower = question.lower()

    # Tecnologias encontradas no currículo
    technologies = []
    experiences = []

    # Mapeamento mais abrangente de tecnologias
    tech_keywords = {
        'python': 'Python',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'fastapi': 'FastAPI',
        'postgresql': 'PostgreSQL',
        'postgres': 'PostgreSQL',
        'mysql': 'MySQL',
        'oracle': 'Oracle',
        'selenium': 'Selenium',
        'matplotlib': 'Matplotlib',
        'whisper': 'Whisper',
        'openai': 'OpenAI',
        'api openal': 'OpenAI',
        'n8n': 'N8N',
        'django': 'Django',
        'flask': 'Flask',
        'react': 'React',
        'node': 'Node.js',
        'mongodb': 'MongoDB',
        'redis': 'Redis',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'git': 'Git',
        'aws': 'AWS',
        'azure': 'Azure',
        'gcp': 'Google Cloud'
    }

    # Buscar tecnologias
    for keyword, tech_name in tech_keywords.items():
        if keyword in context_lower:
            technologies.append(tech_name)

    # Buscar experiências/cargos
    job_keywords = {
        'desenvolvedor': 'desenvolvedor',
        'developer': 'desenvolvedor',
        'programador': 'programador',
        'analista': 'analista',
        'engenheiro': 'engenheiro de software',
        'tech lead': 'tech lead',
        'arquiteto': 'arquiteto de software',
        'full stack': 'desenvolvedor full stack',
        'backend': 'desenvolvedor backend',
        'frontend': 'desenvolvedor frontend'
    }

    for keyword, job_name in job_keywords.items():
        if keyword in context_lower:
            experiences.append(job_name)

    # Verificar tipo de pergunta e gerar resposta apropriada
    if any(word in question_lower for word in ['desenvolvedor', 'developer', 'software', 'programação', 'backend', 'frontend', 'programador']):  # noqa: E501
        if technologies and experiences:
            tech_list = ', '.join(technologies[:5])
            exp_list = ', '.join(set(experiences))
            return {
                "answer": "Sim",
                "justification": f"O candidato possui experiência como {exp_list} e domina as seguintes tecnologias: {tech_list}."  # noqa: E501
            }
        elif technologies:
            tech_list = ', '.join(technologies[:5])
            return {
                "answer": "Sim",
                "justification": f"O candidato possui conhecimento técnico em: {tech_list}."  # noqa: E501
            }
        elif experiences:
            exp_list = ', '.join(set(experiences))
            return {
                "answer": "Sim",
                "justification": f"O candidato possui experiência como {exp_list}."  # noqa: E501
            }
        else:
            return {
                "answer": "Não",
                "justification": "O currículo não apresenta experiências específicas em desenvolvimento de software ou tecnologias relacionadas."  # noqa: E501
            }

    # Para outras perguntas, análise mais genérica
    if technologies or experiences:
        skills = []
        if technologies:
            skills.append(f"tecnologias: {', '.join(technologies[:5])}")
        if experiences:
            skills.append(f"experiência como: {', '.join(set(experiences))}")

        justification = f"O candidato possui {' e '.join(skills)}."
        return {
            "answer": "Sim",
            "justification": justification
        }

    return {
        "answer": "Não",
        "justification": "Não foram identificadas competências específicas no currículo para atender à pergunta."  # noqa: E501
    }


def answer_question(context: str, question: str):
    """Executa análise principal: pergunta baseada no conteúdo do currículo"""
    if not context.strip() or len(context.strip()) < 20:
        return {
            "answer": "Não",
            "justification": "Currículo não contém informações suficientes para análise."  # noqa: E501
        }

    # Primeiro, tentar análise direta baseada no contexto
    direct_analysis = fallback_analysis(context, question)

    # Se a análise direta for bem-sucedida e específica, usar ela
    if (direct_analysis["answer"] in ["Sim", "Não"] and
            "tecnologias:" in direct_analysis["justification"]):
        return direct_analysis

    # Caso contrário, tentar o modelo
    prompt = PROMPT_TEMPLATE.format(
        curriculo=context.strip(),
        query=question.strip()
    )

    try:
        result = completion(prompt)
        answer, justification = extract_answer_and_justification(result)

        # Se o modelo retornou algo válido, usar
        if answer in ["Sim", "Não"] and len(justification) > 20:
            return {
                "answer": answer,
                "justification": justification
            }
    except Exception as e:
        print(f"Erro ao usar modelo: {str(e)}")

    # Fallback final: usar análise direta
    return direct_analysis


def process_resumes(resumes_texts: list[str], query: str):
    """Processa múltiplos currículos com a mesma pergunta"""
    return [
        {
            "resume_index": i,
            **answer_question(text, query)
        }
        for i, text in enumerate(resumes_texts)
    ]


def analyze_resume_for_position(resume_text: str, position_query: str):
    """Alias para manter compatibilidade com chamadas externas"""
    return answer_question(resume_text, position_query)
