"""
Demo script to showcase the Diamond Price Predictor animations and features
"""

import streamlit as st
import time

def demo_animations():
    """Demonstrate the various animations in the app"""
    
    st.title("🎬 Diamond Price Predictor - Animation Demo")
    
    st.markdown("""
    ## ✨ Featured Animations & Effects:
    
    ### 🎯 **Main Features:**
    - **Rotating Diamond Logo** - Custom CSS animation with glowing effect
    - **Lottie Animations** - Diamond, loading, and success animations
    - **Navigation Menu** - Horizontal menu with hover effects
    - **Floating Cards** - Smooth floating animation on feature cards
    - **Sparkle Effects** - Diagonal sparkle animation across elements
    - **Progress Bars** - Glowing progress animations
    - **Interactive Charts** - Plotly visualizations with animations
    - **Button Animations** - Pulse and hover effects
    - **Gradient Backgrounds** - Dynamic gradient themes
    
    ### 🎨 **Visual Effects:**
    - **Orbitron Font** - Futuristic Google Font
    - **Glowing Text** - Animated text shadows
    - **Hover Transformations** - Scale and translate effects
    - **Smooth Transitions** - CSS transitions on all elements
    - **Balloons Celebration** - Streamlit balloons on successful predictions
    
    ### 📱 **User Experience:**
    - **Multi-page Navigation** - Home, Predict, About pages
    - **Real-time Feedback** - Loading spinners and progress indicators
    - **Interactive Elements** - Hover effects and click animations
    - **Responsive Design** - Adaptive layout for all screen sizes
    """)
    
    st.markdown("---")
    
    # Animation showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💎 **Diamond Logo Animation**
        ```css
        .diamond-logo {
            animation: rotate 3s linear infinite;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        ```
        """)
    
    with col2:
        st.markdown("""
        ### ✨ **Glow Animation**
        ```css
        .main-title {
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
            to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
        }
        ```
        """)

if __name__ == "__main__":
    demo_animations()
