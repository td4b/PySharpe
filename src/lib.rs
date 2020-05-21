// use yahoo_finance::{ history, Interval};
use std::collections::HashMap;
use itertools::Itertools;

use ndarray::{Array,Axis,Dim,ArrayBase,ViewRepr,IxDynImpl};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_stats::CorrelationExt;

#[macro_use] 
extern crate log;
extern crate simplelog;

extern crate ndarray;

use simplelog::*;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
use numpy::PyArrayDyn;

// #[derive(Clone)]
// struct Stocks {
//    stock: String,
//    timestamp: Vec<String>,
//    close: Vec<f64>,
// }

// https://github.com/fbriden/yahoo-finance-rs/blob/master/src/history.rs

fn permutations(stocks: Vec<String>, n: usize) -> Vec<Vec<String>>  {
    info!("Calculating Possible Permutations in Portfolios.");
    let mut finals = Vec::new();
    let parseable = stocks.iter().combinations(n);
    for combo in parseable {
        // Create Strings from &str
        let mut strings = Vec::new();
        println!("COMBO {:?}",combo);
        for string in combo {
            strings.push(string.to_string())
        }
        finals.push(strings);
    }
    finals
}

// -> HashMap<String, Stocks>
// fn getstocks(stocks: Vec<String>) -> HashMap<String, Stocks> {
//     let mut folio = HashMap::new();


//     for stock in stocks {
//         let data = history::retrieve_interval(&stock, Interval::_5y).unwrap();
//         info!("Got Stock Data for: {}", stock);
//     // Create Data Arrays.
//     let mut time = Vec::new();
//     let mut close = Vec::new();
//     for i in 0..data.len(){
//         time.push(data[i].timestamp.to_string());
//         close.push(data[i].close);
//     }
//     // Generate and Apply Fields & map.
//     folio.insert(stock.to_string(),Stocks {
//         stock: stock,
//         timestamp: time,
//         close: close,
//     });
//     }
//     folio
// }


// Optimization Struct.
#[derive(Debug)]
struct OptStruct {
    stock: String,
    logret: Vec<f64>,
}

// Applies Zeroes to the Opt Struct Data.
// Idea: translate this to standard function for level setting data.
// HashMap<String, ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>>
fn zeroes(map: HashMap<String, OptStruct>) -> HashMap<String, Vec<f64>> {
        let mut folio = HashMap::new();
        let mut vecs = Vec::new();
        for (_, val) in &map {
            vecs.push(val.logret.clone().len());
        }  
        for (key, val) in &map {
            let length = vecs.iter().max().unwrap() - val.logret.clone().len();
            if length != 0 {
                let mut appended = Vec::new();
                appended.append(&mut vec![0.; length]);
                appended.append(&mut val.logret.clone());
                folio.insert(key.to_string(),appended);
            }
            else {
                folio.insert(key.to_string(),val.logret.clone());
            }
        }
        folio
}

// Optimize.
// Main Optimization Function.

fn localcals(stocks: &HashMap<String, ArrayBase<ViewRepr<&mut f64>, Dim<IxDynImpl>>>) 
    -> HashMap<String, OptStruct>{
    // let mut _arr = Array::from(_permutes.clone());
    let mut folio = HashMap::new();

    for (key, data) in stocks {
        let mut log_ret = Vec::new();
        for i in 1..data.len() {
                // log calc.
                log_ret.push((data[i]/data[i-1]).ln());
            }

        // Set normalized Index.
        folio.insert(key.to_string(), OptStruct{
            stock: key.to_string(),
            logret: log_ret,
        });
    }
    folio
}

#[derive(Debug)]
struct FinalStruct {
    stock: String,
    timestamp: Vec<String>,
    weights: Vec<f64>,
    wreturn: Vec<f64>,
    sharpe: Vec<f64>,
}

// finds the max f64 value in a vector returns (max, index).
fn maxed(data: &Vec<f64>) -> (f64,usize) {
    let max = data.iter().cloned().fold(0./0., f64::max);
    let mut index = 0;
    for i in 0..data.len(){
        if data[i] == max {
            index = i;
        }
    }
    (max,index)
}

fn optimize(_portfolio: &Vec<String>, stocks: &HashMap<String, Vec<f64>>) 
         -> (Vec<f64>,f64,f64,f64) {
    
    // info!("Starting Portfolio Optimization For {:?}",_portfolio.clone());
   
    // Main Calculation
    // Random value Generation of weights.
    // do 5000 iterations for the portfolio weights.

    // for returns we need to build a dedicated structure for each _portfolio.
    let mut log_ret_all = Vec::new();
    let mut length = 0;
    let mut shape  = 0;

    for key in _portfolio {
        shape += 1;
        length = stocks.get(&key.to_string()).unwrap().clone().len();
        log_ret_all.push(stocks.get(&key.to_string()).unwrap().clone());
    }
    let mut alldat = Vec::new();
    for ls in log_ret_all {
        for i in ls {
            alldat.push(i);
        }
    }
    let return_matrix = Array::from_shape_vec((shape,length),alldat).unwrap();
    
    // Get the mean log_return Matrix.
    let mut mean_matrix = Vec::new();
    let mut length = 0;
    for ele in return_matrix.axis_iter(Axis(0)){
        length += 1;
        mean_matrix.push(ele.mean().unwrap());
    }
    let mean_matrix = Array::from_shape_vec((1,length),mean_matrix).unwrap();
    
    // // Get the covariance matrix.
    let cov_matrix  = return_matrix.clone().cov(1.).unwrap();
    
    // data storage.
    let mut all_weights = Vec::new();
    let mut rets = Vec::new();
    let mut vol = Vec::new();
    let mut sharpes = Vec::new();

    for _ in 0..5000 {

        // Randomized weight calculations.
        let _seed = 101;
        let mut rng = rand::thread_rng();
        let rands = Array::random_using(_portfolio.len(), Uniform::new_inclusive(0., 1.), &mut rng);
        let mut weights = Vec::new();
        for j in 0..rands.len() {
            weights.push(rands[j]/rands.sum());
        }

        let weightsarr = Array::from(weights.clone());
        let accu_ret = (&mean_matrix * &weightsarr * 252.).sum();
        let mut volatility: f64  = weightsarr.dot(&weightsarr.dot(&(&cov_matrix*252.)));
        volatility = volatility.sqrt();
        let sharpe = &accu_ret/volatility;

        all_weights.push(weights);
        rets.push(accu_ret);
        vol.push(volatility);
        sharpes.push(sharpe);   
     }

    let maxed = maxed(&sharpes);
    (all_weights[maxed.1].clone(), rets[maxed.1], vol[maxed.1], sharpes[maxed.1])
}


#[pyfunction]
/// Formats the sum of two numbers as string
fn getdata(data: &PyDict) 
    -> PyResult<(Vec<Vec<String>>,Vec<Vec<f64>>,Vec<f64>,Vec<f64>,Vec<f64>)> {
     // configure logger
     CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Warn, Config::default(), TerminalMode::Mixed).unwrap(),
            TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed).unwrap(),
        ]
    ).unwrap();

    // convert PyDict & numpy arrays to Rust Structures.
    let mut rustdata = HashMap::new();
    for (key, val) in data.iter() {
       let key = key.extract::<String>().unwrap().to_string();
       let val = val.extract::<&PyArrayDyn<f64>>().unwrap().as_array_mut();
       rustdata.insert(key,val);
    }

    info!("Starting Stock Processing.");
    let mut _stocks = Vec::new();
    for (key, _val) in &rustdata {
        _stocks.push(key.to_string());
    }

    let mut _perms = permutations(_stocks.clone(),6);
    // // // perform local calcs (such as return)
    let _localdata = localcals(&rustdata);  
    // // transform!(_localdata);
    let _folios = zeroes(_localdata);

    let mut portfolios = Vec::new();
    let mut maxweights = Vec::new();
    let mut maxrets = Vec::new();
    let mut maxvol = Vec::new();
    let mut maxsharpes = Vec::new();

    // find max values for all permutations of the _portfolios.
    info!("Testing all Possible Permutations of: {:?} variations.",_perms.len());
    for i in 0.._perms.len() {
        let results = optimize(&_perms[i], &_folios);
        portfolios.push(_perms[i].clone());
        maxweights.push(results.0);
        maxrets.push(results.1);
        maxvol.push(results.2);
        maxsharpes.push(results.3);
    }
    Ok((portfolios,maxweights,maxrets,maxvol,maxsharpes))
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn statsdata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(getdata))?;

    Ok(())
}