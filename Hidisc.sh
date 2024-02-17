#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-359 -p alvis
#SBATCH -t 3-12:00:00
#SBATCH --gpus-per-node=A100:4

####!/bin/bash


$1
