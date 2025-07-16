import pickle
from flask import Flask,request,app,jasonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)