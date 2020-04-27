from django.conf.urls import url
from demo_app import views

urlpatterns = [
    url('home/', views.home, name="home"),
    url('test/', views.test, name="test"),
    url('update/', views.update, name="update"),
    url('uploadImg/', views.uploadImg, name="uploadImg"),
    url('img_rectification/', views.img_rectification, name="img_rectification"),
    url('img_seg/', views.img_seg, name="img_seg"),
    url('img_key/', views.img_key, name="img_key"),
    url('measure/', views.measure, name="measure"),
]