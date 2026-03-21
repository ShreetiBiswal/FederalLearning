export function deepScale(arr, factor) {
    if (Array.isArray(arr)) {
        return arr.map(val => deepScale(val, factor));
    }
    return arr * factor;
}

export function deepAdd(arr1, arr2) {
    if (Array.isArray(arr1)) {
        return arr1.map((val, i) => deepAdd(val, arr2[i]));
    }
    return arr1 + arr2;
}