
# Station excluded from the model for errors or lack of data:
excluded = ['guadalajara', 'pontevedra', 'santiago-compostela']

# Names of all the types of pollen, in the order they appear in pollen_scraped.h5
pollen_names = ['alnus', 'alternaria', 'artemisa', 'betula', 'carex',
                'castanea', 'cupresaceas', 'fraxinus', 'gramineas', 'mercurialis',
                'morus', 'olea', 'palmaceas', 'pinus', 'plantago', 'platanus',
                'populus', 'amarantaceas', 'quercus', 'rumex', 'ulmus', 'urticaceas']

# Names of all the pollen stations, in alphabetical order
pollen_stations = ['albacete', 'alcazar', 'alicante', 'almeria', 'avila', 'badajoz',
         'barcelona', 'barcelona-uab', 'bejar', 'bilbao', 'burgos', 'burjassot', 'caceres',
         'cadiz', 'cartagena', 'castellon-de-la-plana', 'ciudad-real', 'cordoba', 'coruña',
         'cuenca', 'elche', 'gerona', 'granada', 'gijon', 'guadalajara', 'huelva', 'huesca',
         'jaen-hospital', 'jaen', 'jativa', 'las-palmas', 'leon', 'lerida', 'logroño',
          'madrid-subiza', 'madrid-hospital', 'malaga', 'murcia', 'oviedo', 'palencia',
          'palma-mallorca', 'pamplona', 'ponferrada', 'pontevedra', 'salamanca', 'san-sebastian',
          'santa-cruz-tenerife', 'santander', 'santiago-compostela', 'segovia', 'sevilla-macarena',
          'sevilla-tomillar', 'soria', 'talavera', 'tarragona', 'teruel', 'toledo', 'torrelavega',
           'tudela', 'valencia', 'valladolid', 'vitoria', 'zamora', 'zaragoza']

# Dictionary containing the station id of each location in the AEMET API
weather_stations = {'albacete': '8178D', 'alcazar':'4121', 'alicante': '8025', 'almeria': '6325O',
                    'avila': '2444', 'badajoz' : '4452', 'barcelona': '0076', 'barcelona-uab': '0200E',
                    'bejar': '2870', 'bilbao': '1082', 'burgos':'2331', 'burjassot':'8414A',
                    'caceres': '3469A', 'cadiz': '5973', 'cartagena': '7012C', 'castellon-de-la-plana':'8500A',
                   'ciudad-real': '4121', 'cordoba': '5402', 'coruña': '1387', 'cuenca': '8096',
                   'elche': '8019', 'gerona': '0367', 'granada':'5530E', 'gijon': '1208H', 'guadalajara': '3168C',
                   'huelva': '4642E', 'huesca': '9898', 'jaen-hospital': '5270B', 'jaen': '5270B',
                   'jativa': '8293X', 'las-palmas': 'C029O', 'leon': '2661', 'lerida': '9771C', 'logroño':'9170',
                   'madrid-subiza': '3196', 'madrid-hospital': '3194U', 'malaga': '6155A', 'murcia': '7012C',
                   'oviedo': '1249I', 'palencia': '2400E', 'palma-mallorca': 'B278', 'pamplona': '9262',
                   'ponferrada': '1549', 'pontevedra': '1484C', 'salamanca': '2870', 'san-sebastian': '1024E',
                   'santa-cruz-tenerife': 'C449C', 'santander': '1111', 'santiago-compostela': '1475X',
                   'segovia': '2465', 'sevilla-macarena': '5783', 'sevilla-tomillar': '5783', 'soria':'2030',
                   'talavera': '3365A', 'tarragona': '0016A', 'teruel': '8368U', 'toledo': '3260B',
                   'torrelavega': '1109', 'tudela': '9434', 'valencia': '8416', 'valladolid': '2422',
                   'vitoria': '9091O', 'zamora': '2614', 'zaragoza': '9434'}

# URL of the AEMET API
aemet_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/estacion/{}"

# API key for accessing the AEMET API. This should probably be private lol
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJib2xpdG8yaGRAZ21haWwuY29tIiwianRpIjoiMjUyOTY3MDQtZDYzZS00Zjk2LWE3NTktYTY4M" \
          "zE2NzVmMjE0IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE2MDE2NTY1NzQsInVzZXJJZCI6IjI1Mjk2NzA0LWQ2M2UtNGY5Ni1hNzU5LWE2ODM" \
          "xNjc1ZjIxNCIsInJvbGUiOiIifQ.KzMsubyf4Ux1jxAgu5cGKGZ7rUaGYUreYu8AR0isWjM"

# Days in the analysis phase
anal_size = 30
# Days in the prediction phase
pred_size = 10
# Total temporal span of the model
window_size = anal_size + pred_size

# Features of the model(day to day data)
features = ['pollen', 'tmax', 'tmed', 'tmin', 'presMax', 'presMin', 'velmedia', 'racha', 'prec', 'dx', 'dy']
n_features = len(features)
# Time cycles relevant in pollen levels evolution. According to (Nowosad et al. 2015) the most important temporal spans
# are 1 day, 3.5 days and more than 15 days so we will use the analysis size, and then obviously the year(365)
cycles = [365, anal_size, 4]
n_cycles = len(cycles)

n = n_features + 2*n_cycles