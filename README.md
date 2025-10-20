# AeroSense: Real-Time Prediction of Hard Landings in Flights

AeroSense is a cockpit-deployable machine learning system designed to assist flight crews in making timely go-around decisions by predicting the likelihood of a hard landing in real-time. The system integrates a hybrid neural network model trained on real-world flight data to improve aviation safety and decision-making during the approach phase of a flight.

# Project Overview

More than 50% of commercial aviation accidents occur during approach and landing phases, many of which could be avoided by timely go-around decisions. Currently, these decisions rely heavily on pilot judgment under high-pressure conditions.
AeroSense aims to:
<li>Predict the probability of a hard landing before touchdown
<li>Provide real-time decision support to flight crews
<li>Reduce risk by enhancing situational awareness

# Key Features

<li>Hybrid Neural Network Model trained on over 58,000 commercial flight records
<li>Inputs include: actuator positions, pilot control inputs, flight path data, and environmental conditions

# Performance:

<li>Sensitivity: 85% — correctly identifies potential hard landings
<li>Specificity: 74% — avoids unnecessary false alarms
<li>Designed for real-time deployment in cockpit systems

# Dataset

<li>Source: Commercial flight data (anonymized)
<li>Size: 58,177 flight records
  
# Features Used:
<li>Aircraft state parameters
<li>Pilot control inputs
<li>Environmental variables (e.g., wind, visibility)
<li>Landing outcomes (normal vs hard)
