
from flask import Flask, render_template, redirect, url_for, session, request,Response,jsonify
from pymongo import MongoClient
from model1 import *
import re
import random
from mailersend import emails

app = Flask(__name__)
app.secret_key = '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'

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client.exercise_assistant
users_collection = db.users

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_info')
def yoga_info():
    return render_template('yoga_info.html')

@app.route('/workout_info')
def workout_info():
    return render_template('workout_info.html')

@app.route('/about')
def about():
    return render_template('about.html')


global otp
def send_mail(email):
    
    mailer = emails.NewEmail('mlsn.dd57f8d60245a2cb758126aa5e95cc7ede87fc30bdb286d7467b359735330f89')

    # define an empty dict to populate with mail values
    mail_body = {}

    mail_from = {
        "name": "Excercise Assistant",
        "email": "info@trial-k68zxl2175k4j905.mlsender.net",
    }

    recipients = [
        {
            "name": "Test User",
            "email": email,
        }
    ]
    global otp
    otp = random.randint(1000,9999)
    print(otp)
    mailer.set_mail_from(mail_from, mail_body)
    mailer.set_mail_to(recipients, mail_body)
    mailer.set_subject("Email Verification", mail_body)
    mailer.set_html_content(f"OTP for email verification is : {otp}", mail_body)

    # using print() will also return status code and data
    print(mailer.send(mail_body))
    return otp

def get_otp():
    return otp

@app.route('/forgotpassword/',methods=['POST','GET'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    
    elif request.method == 'POST':
        session['email_id'] = request.form.get('emailid')
        send_mail(session['email_id'])
        return render_template('otp_page.html')

@app.route('/verify_email/',methods=['POST'])
def verify_email():
    otp = request.form['otp']
    otp = int(otp)
    if otp == get_otp():
        send_password(session['email_id'])
        return render_template('password_sent.html')
    else:
        return 'Error'

def send_password(email):

    mailer = emails.NewEmail('mlsn.dd57f8d60245a2cb758126aa5e95cc7ede87fc30bdb286d7467b359735330f89')

    # define an empty dict to populate with mail values
    mail_body = {}

    mail_from = {
        "name": "Excercise Assistant",
        "email": "info@trial-k68zxl2175k4j905.mlsender.net",
    }

    recipients = [
        {
            "name": "Test User",
            "email": email,
        }
    ]
   
    mailer.set_mail_from(mail_from, mail_body)
    mailer.set_mail_to(recipients, mail_body)
    mailer.set_subject("Forget Password", mail_body)

    user_data = users_collection.find_one({'emailid': email})
    user_pass = user_data['password']
    mailer.set_html_content(f"Your Password is : {user_pass}", mail_body)

    # using print() will also return status code and data
    print(mailer.send(mail_body))

def get_otp():
    return otp


@app.route('/signup/')
def signup():
    return render_template('signup.html')

@app.route('/signup_validation/',methods=['POST'])
def signup_validation():

    print('inside signup_validation')

    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

    emailid = request.form['emailid']
    password = request.form['password']

    if not re.match(email_regex, emailid):
        return jsonify({'error': 'Please enter a valid email address'}), 400
        
    if not re.match(password_regex, password):
        return jsonify({'error': 'Password must be at least 8 characters long and contain at least 1 uppercase letter, 1 lowercase letter, 1 digit, and 1 special character'}), 400


    # Check if username already exists
    if users_collection.find_one({'emailid': emailid}):
        return jsonify({'error':'EmailID already exists!'}),401

    # Add new user to database
    users_collection.insert_one({'emailid': emailid, 'password': password})
    # return user to login page
    return jsonify({'success':'success'}), 200


@app.errorhandler(404)
def page404(error):
    return render_template('404.html'),404



@app.route('/login/')
def login():
    return render_template('login.html')

@app.route('/login_validation/',methods=['POST'])
def login_validation():
    # if user already login
    if session.get('emailid'):
        return redirect(url_for('index'))

    emailid = request.form['emailid']
    password = request.form['password']

   
    # Check if email id registered or not
    if users_collection.find_one({'emailid':emailid}) == None:
        return jsonify({'error':'Email id not registered'}), 401
    
    # Check if username and password match
    user = users_collection.find_one({'emailid': emailid, 'password': password})
    if user:
        session['emailid'] = emailid
        return jsonify({'success':'success'}), 200

    return jsonify({'error':'Invalid Login credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('emailid', None)
    return redirect(url_for('index'))

@app.route('/explore_yoga')
def explore_yoga():
    if 'emailid' not in session:
        return redirect(url_for('login'))  # Redirect to login if user is not logged in
    return render_template('yoga_info.html')

@app.route('/yoga_do')
def yoga_do():
    if 'emailid' not in session:
        return redirect(url_for('login')) 
    return render_template('yoga_categories.html')

@app.route('/workout')
def workout():
    if 'emailid' not in session:
        return redirect(url_for('login'))  # Redirect to login if user is not logged in
    
    return render_template('workout.html')



# Route for the exercise description page (example for chest exercises)
@app.route('/workout/chest')
def chest_exercises():
    # You can render a specific template for chest exercises description
    return render_template('chest_exercises.html')



@app.route('/chest_exercises')
def demo():
    return render_template('pushup.html')

@app.route('/chest_exercises1')
def demo1():
    return render_template('dumbbell_flys.html')

@app.route('/chest_exercises2')
def demo2():
    return render_template('dumbbell_pullover.html')

@app.route('/chest_exercises3')
def demo3():
    return render_template('flat_benchpress.html')

@app.route('/chest_exercises4')
def demo4():
    return render_template('incline_benchpress.html')

@app.route('/chest_exercises5')
def demo5():
    return render_template('decline_benchpress.html')

@app.route('/chest_exercises6')
def demo6():
    return render_template('alternate_dumbbellpress.html')

@app.route('/back_exercises')
def demo7():
    return render_template('deadlift.html')

@app.route('/back_exercises1')
def demo8():
    return render_template('pullup.html')

@app.route('/back_exercises2')
def demo9():
    return render_template('barbellbentoverrow.html')

@app.route('/back_exercises3')
def demo10():
    return render_template('seatedrows.html')

@app.route('/back_exercises4')
def demo11():
    return render_template('dumbbellbentoverrow.html')

@app.route('/shoulder_exercises')
def demo12():
    return render_template('doubledumbbellshoulderpress.html')

@app.route('/shoulder_exercises1')
def demo13():
    return render_template('shoulderlateralraise.html')

@app.route('/shoulder_exercises2')
def demo14():
    return render_template('shoulderuprightrows.html')

@app.route('/shoulder_exercises3')
def demo15():
    return render_template('alternatefrontraises.html')

@app.route('/tricep_exercises')
def demo16():
    return render_template('tricepcablepushdown.html')

@app.route('/tricep_exercises1')
def demo17():
    return render_template('standingoverheaddumbbell.html')

@app.route('/tricep_exercises2')
def demo18():
    return render_template('armkickback.html')

@app.route('/tricep_exercises3')
def demo19():
    return render_template('benchdips.html')

@app.route('/bicep_exercises')
def demo20():
    return render_template('doublebicep.html')

@app.route('/bicep_exercises1')
def demo21():
    return render_template('singlearmbicep.html')

@app.route('/bicep_exercises2')
def demo22():
    return render_template('barbellbicep.html')

@app.route('/leg_exercises')
def demo23():
    return render_template('squat.html')

@app.route('/leg_exercises1')
def demo24():
    return render_template('lunge.html')

@app.route('/leg_exercises2')
def demo25():
    return render_template('calf.html')

@app.route('/leg_exercises3')
def demo26():
    return render_template('glutealmuscles.html')

@app.route('/bridge_yoga')
def demo27():
    return render_template('bridgeyoga.html')

@app.route('/downwarddog_yoga')
def demo28():
    return render_template('downwarddogyoga.html')

@app.route('/cobra_yoga')
def demo29():
    return render_template('cobrayoga.html')

@app.route('/triangle_yoga')
def demo30():
    return render_template('triangleyoga.html')



@app.route('/stop_video')
def stop_video():
    return render_template('exercise_count.html',counter=get_counter())     #push up

@app.route('/stop_video5')
def stop_video5():
    return render_template('exercise_count5.html',counter=get_counter())    #dumbbell flys

@app.route('/stop_video6')
def stop_video6():
    return render_template('exercise_count6.html',counter=get_counter())    #dumbbell pullover

@app.route('/stop_video7')
def stop_video7():
    return render_template('exercise_count7.html',counter=get_counter())    #flat bench press

@app.route('/stop_video8')
def stop_video8():
    return render_template('exercise_count8.html',counter=get_counter())    #incline bench press

@app.route('/stop_video9')
def stop_video9():
    return render_template('exercise_count9.html',counter=get_counter())    #decline bench press

@app.route('/stop_video10')
def stop_video10():
    return render_template('exercise_count10.html',counter=get_counter())   #alternate dumbbell press

@app.route('/stop_video11')
def stop_video11():
    return render_template('exercise_count11.html',counter=get_counter())   #deadlift

@app.route('/stop_video12')
def stop_video12():
    return render_template('exercise_count12.html',counter=get_counter())   #pull up

@app.route('/stop_video13')
def stop_video13():
    return render_template('exercise_count13.html',counter=get_counter())   #barbell bent over row

@app.route('/stop_video14')
def stop_video14():
    return render_template('exercise_count14.html',counter=get_counter())   #seated row exercise

@app.route('/stop_video15')
def stop_video15():
    return render_template('exercise_count15.html',counter=get_counter())   #dumbbell bent over row

@app.route('/stop_video16')
def stop_video16():
    return render_template('exercise_count16.html',counter=get_counter())   #double dumbbell shoulder press

@app.route('/stop_video17')
def stop_video17():
    return render_template('exercise_count17.html',counter=get_counter())   #shoulder lateral raise

@app.route('/stop_video18')
def stop_video18():
    return render_template('exercise_count18.html',counter=get_counter())   #shoulder upright row

@app.route('/stop_video19')
def stop_video19():
    return render_template('exercise_count19.html',counter=get_counter())   #alternate front raises

@app.route('/stop_video20')
def stop_video20():
    return render_template('exercise_count20.html',counter=get_counter())   #tricep cable pushdown

@app.route('/stop_video21')
def stop_video21():
    return render_template('exercise_count21.html',counter=get_counter())   #standing overhead dumbbell extension

@app.route('/stop_video22')
def stop_video22():
    return render_template('exercise_count22.html',counter=get_counter())   #arm kickback

@app.route('/stop_video23')
def stop_video23():
    return render_template('exercise_count23.html',counter=get_counter())   #bench dips

@app.route('/stop_video24')
def stop_video24():
    return render_template('exercise_count24.html',counter=get_counter())   #double bicep

@app.route('/stop_video25')
def stop_video25():
    return render_template('exercise_count25.html',counter=get_counter())    #single arm bicep

@app.route('/stop_video26')
def stop_video26():
    return render_template('exercise_count26.html',counter=get_counter())   #barbell bicep curl

@app.route('/stop_video27')
def stop_video27():
    return render_template('exercise_count27.html',counter=get_counter())   #squat

@app.route('/stop_video28')
def stop_video28():
    return render_template('exercise_count28.html',counter=get_counter())   #lunge

@app.route('/stop_video29')
def stop_video29():
    return render_template('exercise_count29.html',counter=get_counter())   #calf

@app.route('/stop_video30')
def stop_video30():
    return render_template('exercise_count30.html',counter=get_counter())   #gluteal muscles



@app.route('/stop_video1')
def stop_video1():
    return render_template('exercise_count1.html',counter=get_counter())    #Bridge Pose yoga

@app.route('/stop_video2')
def stop_video2():
    return render_template('exercise_count2.html',counter=get_counter())    #Downwarddog Pose yoga

@app.route('/stop_video3')
def stop_video3():
    return render_template('exercise_count3.html',counter=get_counter())    #Cobra Pose yoga

@app.route('/stop_video4')
def stop_video4():
    return render_template('exercise_count4.html',counter=get_counter())    #Triangle Pose yoga


@app.route('/generate_frame')
def generate_frame():
    return Response(pushup_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/generate_frame1')
def generate_frame1():
    return Response(dumbbellflys_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame2')
def generate_frame2():
    return Response(dumbbellpullover_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame3')
def generate_frame3():
    return Response(flatbenchpress_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame4')
def generate_frame4():
    return Response(inclinebenchpress_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame5')
def generate_frame5():
    return Response(declinebenchpress_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame6')
def generate_frame6():
    return Response(alternatedumbbellpress_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame7')
def generate_frame7():
    return Response(deadlift_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame8')
def generate_frame8():
    return Response(pullup_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame9')
def generate_frame9():
    return Response(barbellbentoverrow_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame10')
def generate_frame10():
    return Response(seatedrows_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame11')
def generate_frame11():
    return Response(dumbbellbentoverrow_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame12')
def generate_frame12():
    return Response(doubledumbbellshoulderpress_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame13')
def generate_frame13():
    return Response(shoulderlateralraise_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame14')
def generate_frame14():
    return Response(shoulderuprightrows_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame15')
def generate_frame15():
    return Response(alternatefrontraises_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame16')
def generate_frame16():
    return Response(tricepscablepushdown_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame17')
def generate_frame17():
    return Response(standingoverheaddumbbell_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame18')
def generate_frame18():
    return Response(armkickback_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame19')
def generate_frame19():
    return Response(benchdips_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame20')
def generate_frame20():
    return Response(doublebicep_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame21')
def generate_frame21():
    return Response(singlearmbicep_exerises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame22')
def generate_frame22():
    return Response(barbellbicep_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame23')
def generate_frame23():
    return Response(squat_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame24')
def generate_frame24():
    return Response(lunge_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame25')
def generate_frame25():
    return Response(calf_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame26')
def generate_frame26():
    return Response(glutealmuscles_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame27')
def generate_frame27():
    return Response(bridgeyoga_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame28')
def generate_frame28():
    return Response(downwarddogpose_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame29')
def generate_frame29():
    return Response(cobrapose_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_frame30')
def generate_frame30():
    return Response(trianglepose_exercises(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/workout/back')
def back_exercises():
    # You can render a specific template for back exercises description
    return render_template('back_exercises.html')

@app.route('/workout/leg')
def leg_exercises():
    # You can render a specific template for leg exercises description
    return render_template('leg_exercises.html')

@app.route('/workout/bicep')
def bicep_exercises():
    # You can render a specific template for bicep exercises description
    return render_template('bicep_exercises.html')

@app.route('/workout/tricep')
def tricep_exercises():
    # You can render a specific template for tricep exercises description
    return render_template('tricep_exercises.html')

@app.route('/workout/shoulder')
def shoulder_exercises():
    # You can render a specific template for shoulder exercises description
    return render_template('shoulder_exercises.html')

if __name__ == '__main__':
    app.run(debug=True)

