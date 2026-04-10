# Stage 1: Build the React Application
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies FIRST for better layer caching
COPY package.json package-lock.json* ./
RUN npm install

# Copy source code and build it
COPY . .
RUN npm run build

# Stage 2: Serve the Built App using Nginx
FROM nginx:alpine

# Copy the built React app from the previous stage
COPY --from=build /app/build /usr/share/nginx/html

# Replace default Nginx config to support client-side routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
