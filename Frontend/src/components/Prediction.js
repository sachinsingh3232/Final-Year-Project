import React, { useState } from "react";
import axios from "axios";
import Axios from 'axios';
import { call_me } from "../helper";
import DoctorCard from "./DoctorCard";
import useTokenCheck from "../helper/tokenCheck";
import { io } from "socket.io-client";
import { useFetchUser } from "../context/userContext";
import Spinner from "./Spinner";
const l1 = [
  "back_pain",
  "constipation",
  "abdominal_pain",
  "diarrhoea",
  "mild_fever",
  "yellow_urine",
  "yellowing_of_eyes",
  "acute_liver_failure",
  "fluid_overload",
  "swelling_of_stomach",
  "swelled_lymph_nodes",
  "malaise",
  "blurred_and_distorted_vision",
  "phlegm",
  "throat_irritation",
  "redness_of_eyes",
  "sinus_pressure",
  "runny_nose",
  "congestion",
  "chest_pain",
  "weakness_in_limbs",
  "fast_heart_rate",
  "pain_during_bowel_movements",
  "pain_in_anal_region",
  "bloody_stool",
  "irritation_in_anus",
  "neck_pain",
  "dizziness",
  "cramps",
  "bruising",
  "obesity",
  "swollen_legs",
  "swollen_blood_vessels",
  "puffy_face_and_eyes",
  "enlarged_thyroid",
  "brittle_nails",
  "swollen_extremeties",
  "excessive_hunger",
  "extra_marital_contacts",
  "drying_and_tingling_lips",
  "slurred_speech",
  "knee_pain",
  "hip_joint_pain",
  "muscle_weakness",
  "stiff_neck",
  "swelling_joints",
  "movement_stiffness",
  "spinning_movements",
  "loss_of_balance",
  "unsteadiness",
  "weakness_of_one_body_side",
  "loss_of_smell",
  "bladder_discomfort",
  "foul_smell_of urine",
  "continuous_feel_of_urine",
  "passage_of_gases",
  "internal_itching",
  "toxic_look_(typhos)",
  "depression",
  "irritability",
  "muscle_pain",
  "altered_sensorium",
  "red_spots_over_body",
  "belly_pain",
  "abnormal_menstruation",
  "dischromic _patches",
  "watering_from_eyes",
  "increased_appetite",
  "polyuria",
  "family_history",
  "mucoid_sputum",
  "rusty_sputum",
  "lack_of_concentration",
  "visual_disturbances",
  "receiving_blood_transfusion",
  "receiving_unsterile_injections",
  "coma",
  "stomach_bleeding",
  "distention_of_abdomen",
  "history_of_alcohol_consumption",
  "fluid_overload",
  "blood_in_sputum",
  "prominent_veins_on_calf",
  "palpitations",
  "painful_walking",
  "pus_filled_pimples",
  "blackheads",
  "scurring",
  "skin_peeling",
  "silver_like_dusting",
  "small_dents_in_nails",
  "inflammatory_nails",
  "blister",
  "red_sore_around_nose",
  "yellow_crust_ooze",
];
l1.sort();

function Prediction() {
  useTokenCheck(); // token check
  // eslint-disable-next-line 
  const [socket, setSocket] = useState(null);
  //online doctor list
  const [onlineDoc, setOnlineDoc] = useState({
    data: [],
    isPending: true,
    error: null,
  });
  const [spec, setSpec] = useState([]);
  const [sym1, setSym1] = useState("");
  const [sym2, setSym2] = useState("");
  const [sym3, setSym3] = useState("");
  const [sym4, setSym4] = useState("");
  const [resp, setResp] = useState([]);
  const [wait, setWait] = useState(false);
  const { state } = useFetchUser(); // User data

  const submitHandler = async (e) => {
    e.preventDefault();
    try {
      // alert("Please Wait!")
      setWait(true)
      setResp([]);
      const response = await axios.post(`http://localhost:8000/receive_data`, {
        name: state?.data?.name ? state?.data?.name : "PatientName",
        sym1,
        sym2,
        sym3,
        sym4,
      }, { withCredentials: true });
      setWait(false)
      setResp(response?.data?.diseases);
      if (response?.data?.diseases.length > 0) {
        setSpec(call_me(response?.data?.diseases));
        axios
          .post(`${process.env.REACT_APP_API_URL}/api/v1/doctor/getSpecializedDoctors`,
            {
              specializations: call_me(response?.data?.diseases)
            },
            {
              headers: {
                'x-acess-token': localStorage.getItem('token'),
              }
            }
          )
          .then((res) => {
            const newSocket = io(`${process.env.REACT_APP_API_URL}/`); // socket connect
            setSocket(newSocket);
            getOnlineDoc(newSocket, setOnlineDoc, spec);
          })
          .catch((err) => {
            setWait(false)
            console.log(err.response.data);
          });
      }
    } catch (e) {
      setWait(false);
      console.log(e);
    }
  };
  return (
    <div>
      <div className="flex items-center justify-center">
        <form onSubmit={submitHandler} className="flex flex-col w-1/2 items-center justify-center  pt-10">
          <select required onChange={(e) => setSym1(e.target.value)} className='block w-1/2 p-2 mb-5 appearance-none pl-4 pr-3 py-2 rounded-lg border-2 border-gray-200 outline-none focus:border-indigo-500'>
            <option value="">Select Symptom1</option>
            {l1.map((opt) => (
              <option value={opt}>{opt}</option>
            ))}
          </select>
          <select required onChange={(e) => setSym2(e.target.value)} className='block w-1/2 p-2 mb-5 appearance-none pl-4 pr-3 py-2 rounded-lg border-2 border-gray-200 outline-none focus:border-indigo-500'>
            <option value="">Select Symptom2</option>
            {l1.map((opt) => (
              <option value={opt}>{opt}</option>
            ))}
          </select>
          <select onChange={(e) => setSym3(e.target.value)} className='block w-1/2 p-2 mb-5 appearance-none pl-4 pr-3 py-2 rounded-lg border-2 border-gray-200 outline-none focus:border-indigo-500'>
            <option value="">Select Symptom3</option>
            {l1.map((opt) => (
              <option value={opt}>{opt}</option>
            ))}
          </select>
          <select onChange={(e) => setSym4(e.target.value)} className='block w-1/2 p-2 mb-5 appearance-none pl-4 pr-3 py-2 rounded-lg border-2 border-gray-200 outline-none focus:border-indigo-500'>
            <option value="">Select Symptom4</option>
            {l1.map((opt) => (
              <option value={opt}>{opt}</option>
            ))}
          </select>
          <button
            type='submit'
            className='align-middle bg-blue-300 hover:bg-blue-400 text-center px-4 py-2 text-white text-sm font-medium rounded-xl inline-block shadow-lg w-1/2'
          >
            Submit
          </button>
        </form>
        {resp.length > 0 ? (
          <div className="w-400 sm:w-500 md:w-600 lg:w-700 xl:w-800 flex flex-col items-center justify-center ">
            {resp.map((r) => (
              <div className="mt-3 bg-blue-500 flex justify-center font-semibold  w-full p-2 appearance-none pl-4 pr-3 py-2 rounded-lg border-2 border-gray-200 outline-none focus:border-indigo-500">
                {r}
              </div>
            ))}
          </div>
        ) : (
          wait && <Spinner />
        )}
      </div>
      {(!onlineDoc.isPending && onlineDoc.data !== null && onlineDoc.data.length === 0) ?
        <div className='flex justify-center items-center mt-10'>
          <h1 className='font-fontPro text-4xl text-gray-700'>There is no online doctor at this moment.</h1>
        </div>
        :
        <div className='antialiased flex flex-col mt-10'>
          <div className='my-10 mx-auto max-w-7xl w-full px-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-7'>

            {!onlineDoc.isPending && onlineDoc.data !== null ? (
              onlineDoc.data.map((data) => (
                <DoctorCard key={data.name} doctor={data} />
              ))
            ) : (
              <div></div>
            )}
          </div>
        </div>
      }
    </div>
  );
}

const getOnlineDoc = (socket, setOnlineDoc, spec) => {
  socket.on('connect', () => {
    socket.emit('get-online-doctor', socket.id);
  });
  socket.on('updateDoctorList', (doctor) => {
    if (Object.keys(doctor).length === 0) {
      setOnlineDoc({
        data: [],
        isPending: false,
        error: 'No doctor Online',
      });
    } else {
      fetchDoctorData(Object.keys(doctor), setOnlineDoc, spec);
    }
    disconnectSocket(socket);
  });
};

// get doctor data by id
const fetchDoctorData = (doctorId, setOnlineDoc, spec) => {
  const id = doctorId.toString();
  console.log(id)
  const fetchDoctor = async () => {
    try {
      let res = await Axios.get(
        `${process.env.REACT_APP_API_URL}/api/v1/doctor/${id}`,
        {
          headers: {
            'x-acess-token': localStorage.getItem('token'),
          },
        }
      );
      let data = res.data.data;

      if (!Array.isArray(data)) {
        data = [data];
      }
      //filter by specialization
      let fDoc = data.filter(function (el) {
        return spec.includes(el.specialization.specialization);
      });
      data = fDoc;

      setOnlineDoc({
        data: data,
        isPending: false,
        error: null,
      });
    } catch (error) {
      console.log('err')
      setOnlineDoc({
        data: [],
        isPending: false,
        error: error,
      });
    }
  };
  fetchDoctor();
};

const disconnectSocket = (socket) => {
  socket.disconnect();
};

export default Prediction;