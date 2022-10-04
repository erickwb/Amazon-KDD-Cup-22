# Amazon-KDD-Cup-22

TAREFA 1: CLASSIFICAÇÃO DE PRODUTOS DE CONSULTA

Dada uma consulta especificada pelo usuário e uma lista de produtos correspondentes, o objetivo desta tarefa é classificar os produtos para que os produtos relevantes sejam classificados acima dos não relevantes. Isso é semelhante às tarefas padrão de recuperação de informações, mas especificamente no contexto de pesquisa de produtos no comércio eletrônico. A entrada para esta tarefa será uma lista de consultas com seus identificadores. O sistema terá que gerar um arquivo CSV onde o query_id estará na primeira coluna e o  product_id na segunda coluna, onde para cada query_id primeira linha será o produto mais relevante e a última linha o produto menos relevante. Os dados de entrada para cada consulta serão classificados com base em Exatos, Substitutos, Elogios e irrelevantes. No exemplo a seguir para query_1, product_50é o item mais relevante eproduct_80 é o item menos relevante.

