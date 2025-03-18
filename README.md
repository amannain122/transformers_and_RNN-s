🚀 Installation & Setup

1️⃣ Build the Docker Image

Run the following command to build the Docker image:

docker build --no-cache -t sentiment-docker .

2️⃣ Run the Docker Container

If you are using just Docker, run the container with the following command:

docker run -p 5000:5000 sentiment-docker

3️⃣ Build and Run with Docker Compose
You can also use Docker Compose to build and run the application with a single command:

docker-compose up --build

4️⃣ Access the Web Application

Once the container is running, open your browser and visit:

http://localhost:5000

