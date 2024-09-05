import streamlit as st


def show_about():
    st.title("About Chemical Image Classifier")

    st.write("""
    **Chemical Image Classifier** is a tool developed to classify chemical structures from images using advanced machine learning models.

    ### Author
    **Dr. Aleksei Krasnov**  
    Digital Science GmbH (OntoChem)  
    Email: [a.krasnov@digital-science.com](mailto:a.krasnov@digital-science.com)

    ### Address
    Digital Science GmbH (OntoChem)  
    Blücherstrasse 24  
    06120 Halle (Saale)  
    Germany

    ### Citation
    - A. Krasnov, S. Barnabas, T. Böhme, S. Boyer, L. Weber, *Comparing software tools for optical chemical structure recognition*, Digital Discovery (2024). [https://doi.org/10.1039/D3DD00228D](https://doi.org/10.1039/D3DD00228D)
    - L. Weber, A. Krasnov, S. Barnabas, T. Böhme, S. Boyer, *Comparing Optical Chemical Structure Recognition Tools*, ChemRxiv. (2023). [https://doi.org/10.26434/chemrxiv-2023-d6kmg-v2](https://doi.org/10.26434/chemrxiv-2023-d6kmg-v2)

    ### References
    - A. Krasnov, *Images dataset for Chemical Images Classifier model*. [https://zenodo.org/records/13378718](https://zenodo.org/records/13378718)
    - A. Krasnov, *Chemical Image Classifier Model*. [https://zenodo.org/records/10709886](https://zenodo.org/records/10709886)
    """)
