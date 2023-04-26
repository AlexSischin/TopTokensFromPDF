import os
import pdfplumber


def find_files_recursively(directory):
    return [os.path.join(root, f) for (root, _, fs) in os.walk(directory) for f in fs]


def extract_texts(pdf_file_names):
    file_texts = []
    for file_name in pdf_file_names:
        page_texts = []
        with pdfplumber.open(file_name) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                page_texts.append(page_text)
        file_texts.append(page_texts)
    return file_texts


def main():
    directory = 'pdf_files'
    file_names = find_files_recursively(directory)
    pdf_file_names = [f for f in file_names if f.endswith('.pdf')]
    texts = extract_texts(pdf_file_names)
    texts = '\n'.join([t for ts in texts for t in ts])
    print(texts)


if __name__ == '__main__':
    main()
