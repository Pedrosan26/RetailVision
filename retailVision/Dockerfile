# Dockerfile
#
# Container image for the React dashboard.
#
# Vite compiles to static files, so the runtime stage is just nginx
# serving them -- Node is a build-time dependency only and never ships.
#
# VITE_API_BASE_URL is a build argument rather than a runtime environment
# variable because Vite inlines env values into the bundle at build time;
# there is no process left to read them once nginx is serving. The
# default suits local compose, where the browser runs on the host and
# reaches the server on its published port.

FROM node:22-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
