FROM tensorflow/tensorflow:2.2.0
MAINTAINER bllweasley20092@gmail.com

RUN apt-get update && \
 apt-get -y install -qq python3-pip cron

COPY vehicle.py ctc_best.h5 requirements.txt /app/vehicle_logging/
RUN python3 -m pip install -r /app/vehicle_logging/requirements.txt

RUN chmod 0744 /app/vehicle_logging/vehicle.py

RUN touch /var/log/cron.log

RUN (crontab -l ; echo "0 8,15 * * * PYTHONIOENCODING=utf-8 python3 /app/vehicle_logging/vehicle.py >> /var/log/cron.log") | crontab

CMD cron && tail -f /var/log/cron.log



