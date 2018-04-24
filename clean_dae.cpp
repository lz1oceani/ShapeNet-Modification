#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <set>
#include <map>
using namespace std;

typedef vector < int > VI;
#define MAX_V 10000000
bool small_triangle[MAX_V];
VI vadj[MAX_V];
int front[MAX_V][3];
int new_len_face[MAX_V];
vector<pair<int,int>> edge[MAX_V];
int in_du[MAX_V], out_du[MAX_V];
set < int > keys_in, keys_out;
VI line;

typedef long double ld;
#define SQR(_) (((ld)(_)) * (_))
#define LD(_) (ld(_))
bool sign;

struct Vector3D
{
	double x, y, z;
	Vector3D(){}
	Vector3D(double x, double y, double z) : x(x), y(y), z(z){}
	Vector3D operator + (const Vector3D &b) const { return Vector3D(x + b.x, y + b.y, z + b.z); }
	void operator += (const Vector3D &b) { x += b.x, y += b.y, z += b.z; }
	Vector3D operator - (const Vector3D &b) const { return Vector3D(x - b.x, y - b.y, z - b.z); }
	Vector3D operator * (const double &b) const { return Vector3D(x * b, y * b, z * b); }
	Vector3D cross(const Vector3D &b) const { return Vector3D(LD(y) * b.z - LD(z) * b.y, LD(z) * b.x - LD(x) * b.z, LD(x) * b.y - LD(y) * b.x); }
	double dot(const Vector3D &b) const { return LD(x) * b.x + LD(y) * b.y + LD(z) * b.z; }
	double len(void) { return sqrt(SQR(x) + SQR(y) + SQR(z)); }
	void normalize(void) { double l = len(); if (l > 1E-6) x /= l, y /= l, z /= l; }
};


inline void add(int x, int y, int id)
{
    edge[x].push_back(make_pair(y, id));
    in_du[y]++, out_du[x]++, keys_in.insert(y), keys_out.insert(x);
}

Vector3D calcnormal(VI &ls, int vertex_index[][3], Vector3D *normal, int normal_index[][3], int v)
{
    Vector3D sumn(0, 0, 0);
    for (auto tri_index : ls)
        for (int i = 0; i < 3; i++)
            if (vertex_index[tri_index][i] == v)
            {
                sumn += normal[normal_index[tri_index][i]];
                break;
            }
    sumn.normalize();
    return sumn;
}

pair <VI, VI> decide_front(int v, VI &adj, int vertex_index[][3], Vector3D *normal, int normal_index[][3])
{
    for (auto x: keys_out) edge[x].clear(), out_du[x] = 0;
    for (auto x: keys_in) in_du[x] = 0;
    keys_in.clear();
    keys_out.clear();

    if (adj.size() == 2)
    {
        VI A, B;
        A.push_back(adj[0]);
        B.push_back(adj[1]);
        return make_pair(A, B);
    }
    vector<VI> A, B;


    for (auto tri_index : adj)
    {
        int a = vertex_index[tri_index][0], b = vertex_index[tri_index][1], c = vertex_index[tri_index][2];
        if (v == a) add(b, c, tri_index);
        else if (v == b) add(c, a, tri_index);
        else add(a, b, tri_index);

    }

    int T = 0;
    map <int, int> mark;
    while (true)
    {
        T += 1;
        int start = 0, minvalue = 100000;

        for (k : keys_out)
        {
            int in_degree_k = in_du[k], out_degree_k = out_du[k];
            if (in_degree_k < minvalue && out_degree_k > 0)
                minvalue = in_degree_k, start = k;
        }
        if (minvalue == 100000) break;
        int x = start;
        VI ls0, ls1, line;
        set<int> incircle;

        Vector3D last_normal, current_normal;
        bool have_last = false;
        while(true)
        {
            incircle.insert(x);
            line.push_back(x);
            if (edge[x].size() == 0) break;
            bool flag = false;
            for (auto pa : edge[x])
            {
                int y = pa.first, tri_index = pa.second;
                if (incircle.find(y) == incircle.end() && mark.find(tri_index) == mark.end())
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (vertex_index[tri_index][j] == v)
                        {
                            current_normal = normal[normal_index[tri_index][j]];
                            current_normal.normalize();
                            break;
                        }
                    }

                    if (!have_last)
                        have_last = true, last_normal = current_normal;
                    if (last_normal.dot(current_normal) < -0.2)
                        continue;
                    mark[tri_index] = T;
                    ls0.push_back(tri_index);
                    in_du[y]--, out_du[x]--;
                    x = y;
                    flag = true;
                    break;
                }
            }
            if (!flag) break;

        }
        bool iscircle = false;
        if (edge[x].size() > 0 && line.size() > 2)
        {
            for (auto pa : edge[x])
            {
                int y = pa.first, tri_index = pa.second;
                if (y == start && mark.find(tri_index) == mark.end())
                {
                    mark[tri_index] = T;
                    ls0.push_back(tri_index);
                    in_du[y] -= 1;
                    out_du[x] -= 1;
                    iscircle = true;
                    break;
                }
            }
        }
        if (iscircle) line.push_back(start);
        reverse(line.begin(), line.end());
        bool onlyone = false;
        for (int i = 0; i < line.size() - 1; i++)
        {
            auto x = line[i];
            if (edge[x].size() == 0)
            {
                onlyone = true;
                break;
            }
            bool flag = false;
            for (auto pa : edge[x])
            {
                int y = pa.first, tri_index = pa.second;
                if (y == line[i + 1] && mark.find(tri_index) == mark.end())
                {
                    mark[tri_index] = T;
                    ls1.push_back(tri_index);
                    in_du[y] -= 1;
                    out_du[x] -= 1;

                    flag = true;
                    break;
                }
            }

            if (!flag)
            {
                onlyone = true;
                break;
            }
        }


        if (onlyone)
        {
            for (auto tri_index : adj)
                if (mark.find(tri_index) == mark.end())
                    ls1.push_back(tri_index);

            return make_pair(ls0, ls1);
        }
        else
        {
            A.push_back(ls0);
            B.push_back(ls1);
        }
    }

    auto nlist = make_pair(A[0], B[0]);
    Vector3D check_normal[2] = {calcnormal(A[0], vertex_index, normal, normal_index, v), calcnormal(B[0], vertex_index, normal, normal_index, v)};
    for (int i = 1; i < A.size(); i++)
    {
        VI &aa = A[i], &bb = B[i];
        Vector3D aan = calcnormal(aa, vertex_index, normal, normal_index, v), bbn = calcnormal(bb, vertex_index, normal, normal_index, v);
        if (aan.dot(check_normal[0]) > 0 && bbn.dot(check_normal[1]) > 0)
        {
            nlist.first.insert(nlist.first.end(), aa.begin(), aa.end());
            nlist.second.insert(nlist.second.end(), bb.begin(), bb.end());
        }
        else if (aan.dot(check_normal[1]) > 0 && bbn.dot(check_normal[0]) > 0)
        {
            nlist.first.insert(nlist.first.end(), bb.begin(), bb.end());
            nlist.second.insert(nlist.second.end(), aa.begin(), aa.end());
        }
    }
    return nlist;
}
FILE *f = fopen("CPP_tmp.txt", "w");

extern "C"
{
    int process(int primitives_num, int vertex_num, int *len_face, int new_index[], Vector3D *vertex, int vertex_index[][3], Vector3D *normal, int normal_index[][3])
    {
        int all_face = 0, new_face = 0;
        for (int i = 1; i < primitives_num; i++)
            len_face[i] += len_face[i - 1];
        all_face = len_face[primitives_num - 1];
        for (int i = 0; i < vertex_num; i++) vadj[i].clear();
        for (int i = 0, j = 0; i < all_face; i++)
        {
            int a = vertex_index[i][0], b = vertex_index[i][1], c = vertex_index[i][2];
            Vector3D n = (vertex[a] - vertex[b]).cross(vertex[a] - vertex[c]);
            small_triangle[i] = (a == b || a == c || b == c || n.len() < 1E-8);
            if (!small_triangle[i])
            {
                vadj[a].push_back(i);
                vadj[b].push_back(i);
                vadj[c].push_back(i);
                new_index[new_face] = i;
                new_face++;
            }
            if (i == len_face[j] - 1) new_len_face[j++] = new_face;
        }
        for (int i = 0; i < primitives_num; i++)
            len_face[i] = new_len_face[i];
        for (int v = 0; v < vertex_num; v++)
        {
            if (vadj[v].size() == 0) continue;
            pair <VI, VI> nlist_x = decide_front(v, vadj[v], vertex_index, normal, normal_index);
            VI nlist[2] = {nlist_x.first, nlist_x.second};

            Vector3D movedir[2];
            memset(movedir, 0, sizeof(movedir));
            for (int i = 0; i < 2; i++)
            {
                for (auto tri_index : nlist[i])
                {

                    int k = 0, *bt = vertex_index[tri_index];
                    if (bt[1] == v) k = 1;
                    else if (bt[2] == v) k = 2;
                    front[tri_index][k] = i;
                    for (int j = 0; j < 3; j++)
                        if (vertex_index[tri_index][j] == v)
                        {
                            movedir[i] += normal[normal_index[tri_index][j]];
                            break;
                        }
                }
                movedir[i].normalize();
            }
            static const double eps = 1 * (1E-4);


            for (int i = 0; i < 2; i++)
            {
                if (movedir[i].len() <= 1E-4)
                    cout << "NOT Change !!!!" << endl;
                vertex[v + i * vertex_num] += movedir[i] * eps;
            }
        }
        for (int i = 0; i < all_face; i++)
            if (!small_triangle[i])
                for (int k = 0; k < 3; k++)
                    vertex_index[i][k] += front[i][k] * vertex_num;
        return new_face;
    }
}
