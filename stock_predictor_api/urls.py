from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index, name='index'),
    path('prediction/<slug:symbol>', views.prediction, name='prediction'),
    path('clear_predictions/', views.clear_predictions, name='clear_predictions'),
    path('get_weights/<str:tickers>', views.calculate_portfolio_weights, name='portfolio-weights'),
]