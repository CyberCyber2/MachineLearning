# importing google_images_download module
from google_images_download import google_images_download
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

options = Options()
options.binary_location = "/usr/bin/google-chrome"
options.add_argument('--ignore-certificate-errors')
options.add_argument("--headless");
options.add_argument("--disable-dev-shm-usage");
options.add_argument("--no-sandbox");

driver = webdriver.Chrome(executable_path = r'/home/cyber/Desktop/chromedriver' ,options=options)
# creating object
response = google_images_download.googleimagesdownload()

search_queries = [ "Search Query Here ]
	



def downloadimages(query):

	arguments = {"keywords": query,
				"limit":2500,
				"print_urls":True,
				"chromedriver":"/home/cyber/Desktop/chromedriver"}
	try:
		response.download(arguments)
	
	# Handling File NotFound Error	
	except FileNotFoundError:
		arguments = {"keywords": query,
					"format": "jpg",
					"limit":4000,
					"print_urls":True,
					"size": "medium"}
					
		# Providing arguments for the searched query
		try:
			# Downloading the photos based
			# on the given arguments
			response.download(arguments)
		except:
			pass

# Driver Code
for query in search_queries:
	downloadimages(query)
	print()
