# Use an official Anaconda base image
FROM continuumio/anaconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Update Conda, create the environment with Python >= 3.9, and install dependencies
RUN conda install -n base conda && \
    conda create --name streamlit_env python=3.10 -y && \
    /bin/bash -c "source activate streamlit_env && pip install -r requirements.txt"

# Expose the default Streamlit port
EXPOSE 8501

# Start Streamlit using the environment and app
CMD ["/bin/bash", "-c", "source activate streamlit_env && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
