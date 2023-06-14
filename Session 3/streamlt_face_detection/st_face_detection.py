import os
import streamlit as st
import matplotlib.pyplot as plt
import face_recognition
import cv2

def process_images(folder_path):
    output_images = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output_images.append(image)

    return output_images

def main():
    st.title("Face Detection App")

    folder_path = st.text_input("Enter the path to the folder containing the images:")

    if st.button("Process"):
        if not folder_path:
            st.error("Please enter a valid folder path.")
        elif not os.path.isdir(folder_path):
            st.error("Invalid folder path. Please make sure it exists.")
        else:
            output_images = process_images(folder_path)
            num_images = len(output_images)
            columns = 3
            rows = (num_images + columns - 1) // columns

            fig_width = columns * 4
            fig_height = rows * 4

            fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height))

            for i, ax in enumerate(axes.flat):
                if i < num_images:
                    ax.imshow(output_images[i])
                    ax.axis("off")
                else:
                    ax.axis("off")

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
