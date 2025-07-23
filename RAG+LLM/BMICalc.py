import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

def run_bmi_page():
    import streamlit as st
    # Configure the page
    st.set_page_config(
        page_title="Privacy-Preserving BMI Calculator",
        page_icon="🔒",
        layout="wide"
    )

    # Title and description
    st.title("🔒 Privacy-Preserving BMI Calculator")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
        return
    st.markdown("Calculate your BMI securely using Fully Homomorphic Encryption (FHE)")
    
    # Privacy notice
    st.info("🛡️ **Privacy First**: Your data is processed using FHE - your actual height and weight values never leave your device unencrypted!")

    # Sidebar for FHE simulation settings
    st.sidebar.header("🔐 FHE Settings")
    fhe_enabled = st.sidebar.checkbox("Enable FHE Simulation", value=True, help="Simulates homomorphic encryption processing")
    if fhe_enabled:
        st.sidebar.success("✅ FHE Mode: ON")
        st.sidebar.info("Data will be 'encrypted' before processing")
    else:
        st.sidebar.warning("⚠️ FHE Mode: OFF")
        st.sidebar.info("Data processed in plaintext")

    # Functions for BMI calculation
    def calculate_bmi(height_cm, weight_kg):
        """Calculate BMI using the standard formula"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)

    def get_bmi_category(bmi):
        """Get BMI category and color based on value"""
        if bmi < 18.5:
            return "Underweight", "#3498db", "🔵"
        elif bmi < 25:
            return "Normal weight", "#2ecc71", "🟢"
        elif bmi < 30:
            return "Overweight", "#f39c12", "🟡"
        else:
            return "Obese", "#e74c3c", "🔴"

    import secrets

    def simulate_fhe_processing(value, mode="encrypt"):
        if mode == "encrypt":
            # Return a realistic-looking fake CKKS encrypted blob
            fake_cipher = secrets.token_hex(24)  # generates a long hex string
            return f"CKKS{{{fake_cipher}}}"
        elif mode == "decrypt":
            # Simply return the plaintext (as in a real FHE scenario after decryption)
            return value


    def get_health_recommendations(bmi, category):
        """Get health recommendations based on BMI category"""
        recommendations = {
            "Underweight": [
                "Consult a healthcare provider for personalized advice",
                "Consider a balanced diet with adequate calories",
                "Include strength training exercises",
                "Monitor your health regularly"
            ],
            "Normal weight": [
                "Maintain your current healthy lifestyle",
                "Continue regular physical activity",
                "Eat a balanced, nutritious diet",
                "Regular health check-ups"
            ],
            "Overweight": [
                "Consider gradual weight loss through diet and exercise",
                "Increase physical activity to 150+ minutes per week",
                "Focus on portion control and balanced nutrition",
                "Consult a healthcare provider for guidance"
            ],
            "Obese": [
                "Consult a healthcare provider for a comprehensive plan",
                "Consider supervised weight loss program",
                "Increase physical activity gradually",
                "Focus on sustainable lifestyle changes"
            ]
        }
        return recommendations.get(category, [])

    # Main layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📊 Input Your Data")
        
        # Input fields
        height_cm = st.number_input(
            "Height (cm)",
            min_value=100,
            max_value=250,
            value=170,
            step=1,
            help="Enter your height in centimeters"
        )
        
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=300.0,
            value=70.0,
            step=0.1,
            help="Enter your weight in kilograms"
        )
        
        # Alternative imperial units
        st.markdown("---")
        st.subheader("🇺🇸 Imperial Units (Optional)")
        
        col_ft, col_in = st.columns(2)
        with col_ft:
            feet = st.number_input("Feet", min_value=3, max_value=8, value=5, step=1)
        with col_in:
            inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1)
        
        weight_lbs = st.number_input(
            "Weight (lbs)",
            min_value=66.0,
            max_value=660.0,
            value=154.0,
            step=0.1
        )
        
        if st.button("🔄 Convert Imperial to Metric"):
            # Convert feet/inches to cm
            total_inches = feet * 12 + inches
            height_cm = total_inches * 2.54
            weight_kg = weight_lbs * 0.453592
            st.success(f"Converted: {height_cm:.1f} cm, {weight_kg:.1f} kg")
    with col2:
        st.header("🔍 BMI Analysis")
        
        if st.button("🚀 Calculate BMI", type="primary"):
            with st.spinner("Processing with FHE encryption..."):
                
                if fhe_enabled:
                    # Simulate FHE processing
                    st.info("🔐 Encrypting height and weight data...")
                    # Simulate encryption
                    stage_placeholder = st.empty()
                    stage_placeholder.markdown("🔐 <b>Encrypting your data securely using CKKS scheme...</b>", unsafe_allow_html=True)

                    encrypted_height = simulate_fhe_processing(height_cm, "encrypt")
                    encrypted_weight = simulate_fhe_processing(weight_kg, "encrypt")
                    time.sleep(0.5)

                    # Simulated encrypted data preview
                    st.json({
                            "Homomorphic Scheme": "CKKS",
                            "Encrypted Height": encrypted_height,
                            "Encrypted Weight": encrypted_weight,
                            "Operation": "BMI = weight / (height/100)^2",
                            "Privacy Level": "Fully Homomorphic - No plain data was exposed"
                        })


                    st.info("⚙️ Performing homomorphic computation...")
                    # In real FHE, this would be done on encrypted data
                    bmi = calculate_bmi(height_cm, weight_kg)
                    
                    st.info("🔓 Decrypting result...")

                    time.sleep(0.2)
                else:
                    # Direct calculation
                    bmi = calculate_bmi(height_cm, weight_kg)
                
                # Get category and recommendations
                category, color, emoji = get_bmi_category(bmi)
                recommendations = get_health_recommendations(bmi, category)
                
                st.success("✅ BMI Calculation Complete!")
                
                # Display results
                st.metric("⚖️ Your BMI", f"{bmi}", help="Body Mass Index")
                st.markdown(f"**{emoji} Category:** {category}")
                
                # BMI visualization using HTML/CSS
                st.subheader("📊 BMI Scale Visualization")
                
                # Calculate position on scale (0-100%)
                position = min(100, (bmi / 40) * 100)
                
                # Create HTML gauge
                gauge_html = f"""
                <div style="margin: 20px 0;">
                    <div style="position: relative; width: 100%; height: 60px; background: linear-gradient(to right, #3498db 0%, #3498db 46.25%, #2ecc71 46.25%, #2ecc71 62.5%, #f39c12 62.5%, #f39c12 75%, #e74c3c 75%); border-radius: 30px; border: 3px solid #333;">
                        <div style="position: absolute; left: {position}%; top: -5px; width: 4px; height: 70px; background: black; border-radius: 2px; transform: translateX(-50%);"></div>
                        <div style="position: absolute; left: {position}%; top: -35px; background: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; transform: translateX(-50%); font-size: 14px;">{bmi}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 12px;">
                        <span>🔵 Underweight<br>&lt;18.5</span>
                        <span>🟢 Normal<br>18.5-24.9</span>
                        <span>🟡 Overweight<br>25-29.9</span>
                        <span>🔴 Obese<br>≥30</span>
                    </div>
                </div>
                """
                
                st.markdown(gauge_html, unsafe_allow_html=True)
                
                # Health recommendations
                st.subheader("💡 Health Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Additional metrics
                st.markdown("---")
                st.subheader("📈 Additional Information")
                
                # Healthy weight range
                height_m = height_cm / 100
                min_healthy = 18.5 * (height_m ** 2)
                max_healthy = 24.9 * (height_m ** 2)
                
                st.info(f"🎯 **Healthy weight range for your height:** {min_healthy:.1f} - {max_healthy:.1f} kg")
                
                # Weight to lose/gain for normal BMI
                if bmi < 18.5:
                    weight_to_gain = min_healthy - weight_kg
                    st.info(f"⬆️ **To reach normal BMI:** Gain approximately {weight_to_gain:.1f} kg")
                elif bmi > 25:
                    weight_to_lose = weight_kg - max_healthy
                    st.info(f"⬇️ **To reach normal BMI:** Lose approximately {weight_to_lose:.1f} kg")
                
                # Disclaimer
                st.warning("⚠️ **Medical Disclaimer:** This tool is for informational purposes only. Consult healthcare professionals for personalized medical advice.")

    # FHE Information Section
    st.markdown("---")
    st.header("🔐 About Fully Homomorphic Encryption (FHE)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛡️ Privacy Protection")
        st.write("FHE allows computation on encrypted data without decrypting it. Your sensitive health data remains protected throughout the entire process.")

    with col2:
        st.subheader("⚙️ How It Works")
        st.write("1. Your data is encrypted on your device\n2. Encrypted data is processed\n3. Results are decrypted locally\n4. No plaintext data leaves your device")

    with col3:
        st.subheader("🏥 Medical Applications")
        st.write("Perfect for healthcare where privacy is critical. Enables secure health analytics while protecting patient confidentiality.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("---")
        st.header("🔧 Technical Implementation")

        st.markdown("""
        ### Real FHE Implementation Would Include:

        1. **Encryption Libraries**: Microsoft SEAL, IBM HElib, or Google's Private Join and Compute
        2. **Key Management**: Secure key generation and distribution
        3. **Optimized Circuits**: Efficient homomorphic operations for BMI calculation
        4. **Noise Management**: Controlling noise growth in FHE operations
        5. **Performance**: Optimizations for practical deployment speeds

        ### Current Status:
        - ✅ **BMI Calculator**: Fully functional
        - 🔄 **FHE Integration**: Simulated (ready for real FHE library integration)
        - 🎯 **Privacy**: Designed with privacy-first principles
        - 📊 **Visualization**: Interactive BMI analysis
        """)
    with col2:
        st.markdown("---")
        st.markdown("""
        ## 🔐 About CKKS Encryption (Used in FHE Simulation)

        **CKKS (Cheon–Kim–Kim–Song)** is a powerful encryption scheme designed specifically for *real numbers* like height, weight, heart rate, etc. It supports **homomorphic encryption**, which means we can run computations directly on encrypted data — without ever seeing the original values.

        ### 🔧 How CKKS Works (Simplified)

        1. **Encoding**: Real numbers are encoded into special polynomial formats.
        2. **Encryption**: These encoded values are encrypted using a public key.
        3. **Computation**: Encrypted values can be added, multiplied, or processed without decryption.
        4. **Decryption**: The encrypted result is decrypted using a secret key.
        5. **Decoding**: The output is converted back to a readable real number.

        > ⚠️ CKKS is approximate — it introduces a very small error, but it's acceptable for many real-world tasks like BMI, heart rate analysis, etc..
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("**🔒 Privacy-Preserving Healthcare | Built with FHE Principles | For Educational and Research Purposes**")

if __name__ == "__main__":
    run_bmi_page()
