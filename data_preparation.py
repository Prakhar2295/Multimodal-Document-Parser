import pdfplumber
import pypdfium2 as pdfium
from datetime import datetime

class data_preparation:
	def __init__(self,filename:str):
		self.filename = filename

	def pdf_to_image(self):
		if self.filename is not None:
			pdf = pdfium.PdfDocument(self.filename)
			page = pdf.get_page(0)
			pil_image = page.render(scale=300 / 25).to_pil()
			timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

			#image_name = f"/pdf_to_img/{self.filename[:-4]}.jpg"
			image_name = f"/pdf_to_img/{self.filename[:-4]}_{timestamp}.jpg"
			pil_image.save(image_name)

	def pdf_searchable(self):
		if self.filename is not None:
			text_list = list()
			with pdfplumber.open(self.filename) as pdf:
				page = pdf.pages[0]
				text = page.extract_text()
				text_list.append(text)
			if text_list >= 1:
				return True
			else:
				return False





