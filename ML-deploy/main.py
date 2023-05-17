from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import PlainTextResponse, HTMLResponse
from starlette_wtf import StarletteForm, CSRFProtectMiddleware, csrf_protect
from wtforms import StringField
from wtforms.validators	import DataRequired
import joblib
import numpy as np
import pandas as pd


app = FastAPI()
  
  
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/', tags=['main route'], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("ml.html", {"request":request})



@app.post('/', response_class=HTMLResponse)
async def predict(request: Request, client_id: int = Form(...), 
                          gender: str = Form(...),
                          age: int = Form(...),
                          maritalstatus: str = Form(...),
                          job: str = Form(...),
                          credit_sum: int = Form(...),
                          credit_month: int = Form(...),
                          education: str = Form(...),
                          living: str = Form(...),
                          monthly_income: int = Form(...)
                          ):
    tariff_id = [1.1 , 1.6 , 1.5 , 1.9 , 1.43, 1.41, 1.32, 1.16, 1.4 , 1.3 , 1.91, 1.44, 1.  ,
                1.17, 1.7 , 1.19, 1.2 , 1.21, 1.23, 1.22, 1.24, 1.94, 1.27, 1.25, 1.28, 1.18, 
                1.26, 1.48, 1.56, 1.29, 1.52] 
    
    tariff=tariff_id[np.random.randint(30)]
    score_shk=round(np.random.rand(),2)+np.random.randint(2)
    credit_count=np.random.randint(20)
    overdue_credit_count=np.random.randint(4)


    list_=[age, credit_count, credit_month, credit_sum, education, 
           gender, job, living, maritalstatus, monthly_income, overdue_credit_count,score_shk,tariff]
    df = pd.DataFrame([list_],columns=['age','credit_count','credit_month','credit_sum',
    'education','gender','job_position','living_region_cleaned',
    'marital_status','monthly_income','overdue_credit_count','score_shk','tariff_id'], )

    enc = joblib.load('encoder.joblib')
    xgb_m = joblib.load('xgb_model.joblib')
    X_enc = enc.fit_transform(df)
    result = xgb_m.predict(X_enc)

    return templates.TemplateResponse("ml.html", {"request": request, "prediction_text":result[0]})






