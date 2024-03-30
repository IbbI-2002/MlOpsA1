# Use an official nginx base image
FROM nginx:alpine

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configuration file into the container
COPY nginx.conf /etc/nginx/conf.d

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static assets over
COPY . /usr/share/nginx/html

# Expose port 80 to the Docker host, so we can access it from the outside.
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
