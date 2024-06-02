{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Monaco;}
{\colortbl;\red255\green255\blue255;\red164\green191\blue255;\red25\green22\blue35;\red255\green255\blue255;
\red252\green115\blue96;\red117\green255\blue242;\red254\green219\blue112;}
{\*\expandedcolortbl;;\cssrgb\c70196\c80000\c100000;\cssrgb\c12941\c11765\c18431;\cssrgb\c100000\c100000\c100000;
\cssrgb\c100000\c53725\c45098;\cssrgb\c51373\c100000\c96078;\cssrgb\c100000\c87843\c51373;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import\cf4 \strokec4  \cf5 \strokec5 pandas\cf4 \strokec4  \cf2 \strokec2 as\cf4 \strokec4  \cf5 \strokec5 pd\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  \cf5 \strokec5 numpy\cf4 \strokec4  \cf2 \strokec2 as\cf4 \strokec4  \cf5 \strokec5 np\cf4 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  \cf5 \strokec5 matplotlib\cf4 \strokec4 .\cf6 \strokec6 pyplot\cf4 \strokec4  \cf2 \strokec2 as\cf4 \strokec4  \cf5 \strokec5 plt\cf4 \strokec4  \cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  \cf5 \strokec5 codecademylib3_seaborn\cf4 \strokec4  \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 coffee\cf4 \strokec4  = \cf5 \strokec5 pd\cf4 \strokec4 .\cf6 \strokec6 read_csv\cf4 \strokec4 (\cf7 \strokec7 'starbucks_customers.csv'\cf4 \strokec4 )\cb1 \
\
\cf5 \cb3 \strokec5 ages\cf4 \strokec4  = \cf5 \strokec5 coffee\cf4 \strokec4 [\cf7 \strokec7 'age'\cf4 \strokec4 ]\cb1 \
\
\cf5 \cb3 \strokec5 min_age\cf4 \strokec4  = \cf5 \strokec5 np\cf4 \strokec4 .\cf6 \strokec6 min\cf4 \strokec4 (\cf5 \strokec5 ages\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 print(\cf5 \strokec5 min_age\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 max_age\cf4 \strokec4  = \cf5 \strokec5 np\cf4 \strokec4 .\cf6 \strokec6 max\cf4 \strokec4 (\cf5 \strokec5 ages\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 print(\cf5 \strokec5 max_age\cf4 \strokec4 )\cb1 \
\
\cb3 print(\cf5 \strokec5 max_age\cf4 \strokec4  - \cf5 \strokec5 min_age\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 mean_age\cf4 \strokec4  = \cf5 \strokec5 np\cf4 \strokec4 .\cf6 \strokec6 mean\cf4 \strokec4 (\cf5 \strokec5 ages\cf4 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 print(\cf5 \strokec5 mean_age\cf4 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 centered_ages\cf4 \strokec4  = \cf5 \strokec5 ages\cf4 \strokec4  - \cf5 \strokec5 mean_age\cf4 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 centered_data\cf4 \strokec4  = \cf5 \strokec5 centered_ages\cf4 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 plt\cf4 \strokec4 .\cf6 \strokec6 hist\cf4 \strokec4 (\cf5 \strokec5 centered_data\cf4 \strokec4 )\cb1 \
\cf5 \cb3 \strokec5 plt\cf4 \strokec4 .\cf6 \strokec6 show\cf4 \strokec4 ()\cb1 \
\
}