# smart-resume-analyzer  
API RESTful desenvolvida com FastAPI para **analisar automaticamente currículos tech** em PDF ou imagem (via OCR), **extrair informações relevantes** e **responder perguntas** usando IA generativa com justificativas.

## Principais tecnologias utilizadas:
- Python 3.12
- FastAPI
- Uvicorn
- EasyOCR
- Transformers
- Torch
- PDF2image
- Docker
- MongoDB

## Objetivo:
- Receber currículos em PDF ou imagem;
- Extrair o texto (usando OCR se necessário);
- Gerar um sumário do conteúdo do currículo;
- Responder a perguntas sobre o currículo com justificativa.

---

## Como executar via Docker:

### 1) Clone o repositório

```bash
git clone https://github.com/seu-usuario/smart-resume-analyzer.git
cd smart-resume-analyzer
```

### 2) Construa a imagem Docker

```bash
docker build -t smart-resume-analyzer .
```

### 3) Execute o container

```bash
docker run --rm -p 8000:8000 smart-resume-analyzer
```

> Isso expõe a API na porta `8000` do seu computador.

---

## Acesse o Swagger:

Abra no navegador:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## Endpoints disponíveis

| Método | Rota           | Descrição                                                                 |
|--------|----------------|--------------------------------------------------------------------------|
| POST   | `/analyze/`    | Recebe um currículo (PDF ou imagem), extrai o texto e responde à pergunta com justificativa |
| GET    | `/`            | Retorna status da API                                                    |

---

## Exemplo de requisição (via Swagger)

Faça um `POST` em `/analyze/` com:

- Arquivo (campo `file`): currículo em PDF ou imagem
- Texto da pergunta (campo `query`): exemplo → *"Essa pessoa tem experiência com Django?"*

---

## Exemplo de resposta:

```json
{
  "summary": "Desenvolvedor Python com experiência em APIs REST e bancos de dados relacionais.",
  "answer": "Sim, o candidato menciona experiência com Django.",
  "justification": "A palavra-chave 'Django' aparece no trecho: 'Experiência com Django, Flask e FastAPI.'"
}
```

---

## Observações

- Se um PDF não tiver texto extraível, o OCR será aplicado automaticamente.
- O modelo de linguagem usado pode ser ajustado no backend (`services/llm.py`).
- O OCR é feito com [EasyOCR](https://github.com/JaidedAI/EasyOCR).

---

## Links úteis

- [FastAPI](https://fastapi.tiangolo.com/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Transformers](https://huggingface.co/docs/transformers)
- [Docker](https://www.docker.com/)

---

## Licença

MIT © 2025