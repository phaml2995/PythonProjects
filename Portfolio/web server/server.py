from flask import Flask,render_template,request, redirect
import csv
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/<string:page_name>')
def page(page_name):
	return render_template(page_name)

def write_to_file(data):
	with open("message.txt", mode = "a") as database:
		email = data["email"]
		subject = data["subject"]
		message = data["message"]
		file = database.write(f"\n{email},{subject},{message}")


def write_to_csv(data):
	with open("database.csv", mode = "a") as database2:
		email = data["email"]
		subject = data["subject"]
		message = data["message"]
		csv_writer = csv.writer(database2, delimiter = ',', quotechar='"',quoting =csv.QUOTE_MINIMAL)
		csv_writer.writerow([email,subject,message])

@app.route('/submit_form', methods=['POST', 'GET'])
def submit_form():
	if request.method == "POST":
		try:
			data = request.form.to_dict()
			write_to_csv(data)
			print(data)
			return redirect('/thankyou.html')
		except:
			return "Error"

	else:
		return "Oh Oh"


# @app.route('/blog')
# def blog():
# 	return "New Blog!"


# @app.route('/about.html')
# def about():
# 	return render_template('about.html')

# @app.route('/project.html')
# def my_project():
# 	return render_template('project.html')


# @app.route('/components.html')
# def componentst():
# 	return render_template('components.html')