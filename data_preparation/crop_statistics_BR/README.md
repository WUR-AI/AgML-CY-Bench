# Brazil subnational crop yield statistics

## Short description
The linked platform allows you to customize and download datasets based on crop, variable, year, spatial aggregation, and region. You can obtain the results in various formats, including CSV, XLSX, TSV, ODS, or HTML, as well as a compressed ZIP file. For larger requests exceeding 200,000 values, you can provide an email address, and once the file is ready, download instructions will be sent to you.

In most cases, the resulting table will have multi-level columns and a header. The structure of the file depends on the information you requested prior to the download and the structure you define (more on this below). Each file contains a "Notas" sheet with supporting information on the data.

For additional information and to avoid redundancy, I refer to the platform itself and the remarks they provide, e.g. at the end of the page and in the downloaded tables.

## Link
https://sidra.ibge.gov.br/tabela/1612

## Publisher
Brazilian Institute of Geography and Statistics (IBGE)

## Dataset owner
Brazilian Institute of Geography and Statistics (IBGE)

## Data card author
Maximilian Zachow

## Dataset overview
The process of obtaining an exemplary wheat dataset (including yield, planted area, harvested area and production) is outlined below. Here, the last four years are selected as an example. 

A translation is given in italic. Basic preprocessing of the downloaded csv files for yield, production, harvested and planted area is given in the notebook `read_and_preprocess_data.ipynb`.

### Selection process via interface on platform 
First, make sure that you adjust the outline at the very top of the page to match the following structure. You can simply drag and drop elements to their correct position:

![test](images/IBGE_layout.png)


 **Variável** (*variable*): 
 - Rendimento médio da produção (Quilogramas por Hectare) - *yield (kg/ha)*
 - Área plantada (Hectares [1988 a 2022]) - *planted area (ha)*
 - Área colhida (Hectares) - *harvested area (ha)* 
 - Quantidade produzida (Toneladas) - *production (t)*

**Produto das lavouras temporárias** (*crop selection*): 
 - Trigo (em grão) - *wheat (grains)*

Other relevant crops and their English translation:
 - Soja (em grão) - *soybeans (grains)*
 - Milho (em grão) - *corn (grains)*
 - Arroz (em casca) - *rice (paddy)*

 **Ano** (*year*):
  - 2022, 2021, 2020, 2019, ... (select as many you want)

**Unidade Territorial** *(territorial unit)*: 
 - Município - *municipality*

Click download and a dialogue opens. Note that the immediate download is only available for up to 200,000 values. On the top of the download window, you will see how many values your current request contains (e.g. * 22.252 valores na seleção - *22,252 values selected*). Instructions on how to obtain requests exceeding 200,000 values are given further below. For now continue with the next steps. Select formato *(format)* CSV (US) and check two boxes "Exibier códigos de territórios *(display territory codes)* and Exibir nomes de territórios *(Display territory names)*. 

**Less than 200,000 values**: Select "Imediato (até 200.000 valores)" *(immediate download)* and click download again. A new tab opens and after ~7 min the download starts.

**More than 200,000 values**: If you selected all available years, e.g. 1974-2022 for wheat, you will end up with ~ 1 Mio. values (too much for the immediate download). In the download dialogue, you can then select A Posteriori (até 3.000.000 valores) *(A Posteriori (up to 3,000,000 values))*. After prodviding a mail-address and clicking download you will receive a message "Gravação a posteriori, A solicitação foi incluída na fila e em breve estará disponível!", indicating that the file will be sent to you via e-mail. In my test case with four variables for wheat and all available years it took around 1.5h until I received the email. In order to download the data from the link in the mail, you need to have a SIDRA account that you can create in less than a minute [here](https://sidra.ibge.gov.br/perfil/usuario/cadastro).  

Regardless of your download methods, save the dataset and proceed to the notebook `read_and_preprocess_data.ipynb` for the preprocessing steps.

## Provenance 

Weekly updates.

## License
Nothing was found on their website regarding data policies. However, IBGE data has been distributed before. See [here](https://figshare.com/articles/dataset/_2008_IBGE_data_for_the_regions_and_states_sampled_in_this_study_/468422?backTo=/collections/The_Genomic_Ancestry_of_Individuals_from_Different_Geographical_Regions_of_Brazil_Is_More_Uniform_Than_Expected/1696325) and [here](https://www.ceicdata.com/en/brazil/sna-2008-gross-domestic-product-per-capita/gross-domestic-product-prices-of-previous-year-ibge)

## Additional information
Not applicable.

### Overview of dataset structure and basic preprocessing

See notebook `read_and_preprocess_data.ipynb`.

## References
IBGE SIDRA. (2022), “Tabela 1612: Área plantada, área colhida, quantidade produzida, rendimento médio e valor da produção das lavouras temporárias”, available at: https://sidra.ibge.gov.br/tabela/1612 (accessed 6 February 2024).
