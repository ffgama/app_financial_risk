import streamlit as st


def main():
    st.write("#### Análise de relatórios com IA")
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    import re
    import os
    import requests

    # Configurações iniciais
    ticker = "HGLG11"  # Altere para o ticker desejado
    url = f"https://www.fundsexplorer.com.br/comunicados?ticker={ticker}"

    # Configurações do Selenium com Chrome em modo headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)

    # Acessa a página
    driver.get(url)

    # Aguarda alguns segundos para garantir que o JavaScript carregue o conteúdo da tabela
    time.sleep(5)  # Pode ser substituído por WebDriverWait para uma espera mais refinada

    # Localiza a tabela que contém os comunicados
    try:
        table = driver.find_element(By.CSS_SELECTOR, "table.default-fiis-table__container__table[data-element='table-comunicados-container']")
    except Exception as e:
        st.write("Tabela principal não encontrada:", e)
        driver.quit()
        exit()

    # Extrai todas as linhas da tabela
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Expressão regular para identificar "Relatório Gerencial - MM/YYYY"
    pattern = re.compile(r"Relat[oó]rio Gerencial\s*-\s*(\d{2})/(\d{4})", re.IGNORECASE)

    # Lista para armazenar os relatórios (mês, ano, link)
    reports = []

    # Percorre as linhas e células procurando a div com a classe 'td_doc'
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        for cell in cells:
            try:
                div_doc = cell.find_element(By.CSS_SELECTOR, "div.td_doc")
                text = div_doc.text.strip()
                match = pattern.search(text)
                if match:
                    mes = match.group(1)
                    ano = match.group(2)
                    # Define o título esperado do link
                    expected_title = f"Download do Comunicado da data {mes}/{ano}"
                    try:
                        # Procura a tag <a> com o atributo title esperado dentro da div
                        a_tag = div_doc.find_element(By.XPATH, f".//a[@title='{expected_title}']")
                        if a_tag:
                            link = a_tag.get_attribute("href")
                            reports.append((mes, ano, link))
                    except Exception as e:
                        st.write("Link não encontrado na div:", e)
            except Exception:
                continue

    driver.quit()

    # Verifica se encontrou algum relatório
    if not reports:
        st.write("Nenhum relatório gerencial encontrado.")
        exit()

    # Cria a pasta para salvar os PDFs, se não existir
    output_folder = "relatorios"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Baixa os PDFs utilizando requests e salva com o nome: ticker_RG_mes_ano.pdf
    headers = {"User-Agent": "Mozilla/5.0"}
    for mes, ano, link in reports:
        pdf_response = requests.get(link, headers=headers)
        if pdf_response.status_code == 200:
            filename = f"{ticker}_RG_{mes}_{ano}.pdf"
            filepath = os.path.join(output_folder, filename)
            with open(filepath, "wb") as f:
                f.write(pdf_response.content)
            st.write(f"PDF salvo: {filepath}")
        else:
            st.write(f"Erro ao baixar o PDF: {link}")


if __name__ == '__main__':
    main()
