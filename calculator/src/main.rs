// Calculator used for calculating linear regression figures

use std::collections::HashMap;

fn main(){
    // TODO: allow for custom data imports. For now its just hardcoded below.
    const DATA: [(i32, i32); 6] = [(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)];

    let pmf = calculate_pmf(&DATA);
    let pmf_x = calculate_marginal_pmf(&DATA, true);
    let pmf_y = calculate_marginal_pmf(&DATA, false);

    let cov = calculate_covariance(&DATA);
    
    let correlation = calculate_correlation_coefficient(&DATA);

    let (b0, b1) = regression_coefficients(&DATA);
    let x = 0.8;
    let y_hat = linear_regression(b0,b1,x as f64);
    
    println!("\nPMF:");
    for(key,value) in &pmf{
        println!("{:?} -> {:.4}", key,value);
    }
    
    println!("\nPMF of X:");
    for (key, value) in &pmf_x {
        println!("fX({}) = -> {:.4}", key, value);
    }

    println!("\nPMF of Y:");
    for (key, value) in &pmf_y {
        println!("fY({}) = -> {:.4}", key, value);
    }

    println!("Cov(X, Y) = {:.2}", cov);

    println!("Correlation Coefficient (ρ) = {:.2}", correlation);

    println!("ŷ = {}+{}x", b0, b1);
    println!("Predicted ŷ for X = {} is {:.2}", x, y_hat);

}

fn calculate_pmf(data: &[(i32,i32)]) ->HashMap<(i32,i32), f64>{
    let mut frequency: HashMap<(i32,i32),i32> = HashMap::new();

    for &pair in data{
        *frequency.entry(pair).or_insert(0)+=1;
    }

    let total_count = data.len() as f64;

    let mut pmf: HashMap<(i32,i32),f64>=HashMap::new();

    for (key, &count) in &frequency{
        pmf.insert(*key, count as f64/total_count);
    }
    pmf
}

fn calculate_marginal_pmf(data: &[(i32,i32)], is_x: bool) -> HashMap<i32, f64>{
    let mut frequency: HashMap<i32, i32> = HashMap::new();

    for &(x,y) in data{
        let value = if is_x {x} else {y};
        *frequency.entry(value).or_insert(0) += 1;
    }

    let total_count = data.len() as f64;
    let mut pmf: HashMap<i32,f64> = HashMap::new();

    for (key, &count) in &frequency {
        pmf.insert(*key, count as f64/ total_count);
    }

    pmf
}

fn calculate_covariance(data: &[(i32,i32)]) -> f64 {
    let (mean_x, mean_y) = calculate_means(data);
    let n = data.len() as f64;

    let exy: f64 = data.iter()
        .map(|(x,y)| (*x as f64) * (*y as f64))
        .sum::<f64>() / n;

    exy - (mean_x * mean_y)

}

fn calculate_means(data: &[(i32,i32)]) -> (f64,f64) {
    let sum_x: i32 = data.iter().map(|(x,_)| *x).sum();
    let sum_y: i32 = data.iter().map(|(_,y)| *y).sum();
    
    let n = data.len() as f64;
    (sum_x as f64 / n, sum_y as f64 /n)
}

fn calculate_mean(data: &[(i32,i32)], is_x: bool) -> f64{
    let sum: i32 = data.iter()
        .map(|(x, y)| if is_x { *x } else { *y })
        .sum();

    sum as f64 / data.len() as f64
}


fn calculate_correlation_coefficient(data: &[(i32,i32)]) -> f64{
    let covariance = calculate_covariance(&data);
    let std_x = calculate_variance(data, true).sqrt();
    let std_y = calculate_variance(data, false).sqrt();
    if std_x == 0.0 || std_y == 0.0 {
        println!("Warning: Standard deviation is zero. Correlation is undefined.");
        return 0.0;
    }

    covariance / (std_x * std_y)
}

// sigma^2
fn calculate_variance(data: &[(i32,i32)], is_x: bool) -> f64{
    let mean = calculate_mean(&data, is_x);

    let ex2 = data.iter()
        .map(|(x, y)| {
            let value = if is_x { *x as f64 } else { *y as f64 };
            value.powi(2)
        })
        .sum::<f64>() / data.len() as f64;

    ex2 - mean.powi(2)
}

// b0, b1
fn regression_coefficients(data: &[(i32,i32)]) -> (f64, f64){
    let cov_xy = calculate_covariance(data);
    let std_x = calculate_variance(data, true);

    let b1 = cov_xy / std_x;

    let (mean_x, mean_y) = calculate_means(data);

    let b0 = mean_y - b1 * mean_x;

    (b0,b1)
}

// Y-hat
fn linear_regression(b0: f64, b1:f64, x: f64)-> f64{
    b0 + b1 * (x)
}