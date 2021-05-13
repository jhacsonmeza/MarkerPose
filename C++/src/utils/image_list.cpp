#include "utils.hpp"

#include <vector>
#include <string>
#include <sstream> // std::istringstream
#include <filesystem>


// Taken from https://stackoverflow.com/a/9745132/11920165
// NOTE: this is not a efficient implementation
bool compareNat(const std::string& a, const std::string& b)
{
    if (a.empty())
        return true;
    if (b.empty())
        return false;
    if (isdigit(a[0]) && !isdigit(b[0]))
        return true;
    if (!isdigit(a[0]) && isdigit(b[0]))
        return false;
    if (!isdigit(a[0]) && !isdigit(b[0]))
    {
        if (toupper(a[0]) == toupper(b[0]))
            return compareNat(a.substr(1), b.substr(1));
        return (toupper(a[0]) < toupper(b[0]));
    }

    // Both strings begin with digit --> parse both numbers
    std::istringstream issa(a);
    std::istringstream issb(b);
    int ia, ib;
    issa >> ia;
    issb >> ib;
    if (ia != ib)
        return ia < ib;

    // Numbers are the same --> remove numbers and recurse
    std::string anew, bnew;
    std::getline(issa, anew);
    std::getline(issb, bnew);
    return (compareNat(anew, bnew));
}

std::vector<std::string> imlist(const std::filesystem::path& folder_path)
{
    std::vector<std::string> files;
    for (const auto& p : std::filesystem::directory_iterator(folder_path))
        files.push_back(p.path().string());

    std::sort(files.begin(), files.end(), compareNat);

    return files;
}