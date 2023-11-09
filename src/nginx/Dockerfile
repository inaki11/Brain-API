FROM nginx:1.21-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d
COPY nginx.crt /etc/nginx/nginx.crt
COPY nginx.key /etc/nginx/nginx.key
COPY self-signed.conf /etc/nginx/self-signed.conf