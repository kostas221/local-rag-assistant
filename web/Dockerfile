# Δύο στάδια: το Node χρειάζεται μόνο για το build· η εικόνα που τρέχει έχει μόνο nginx και τα στατικά αρχεία.

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# nginx χωρίς root, στην 8080 (όπως ο backend τρέχει ως appuser).
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
